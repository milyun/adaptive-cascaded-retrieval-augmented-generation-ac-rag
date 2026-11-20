#!/usr/bin/env python3
"""
E11 Sensitivity Analysis — AC-RAG Heuristic Parameters
=======================================================
Reviewer E11 asked for a sensitivity analysis of:
  1. Demographic similarity weights (w_age / w_gender, Equation 2)
  2. Age-similarity bin thresholds (age cutoffs in the similarity function)
  3. Adaptive semantic relevance thresholds (τ_hybrid=0.75, τ_cross=0.55)

Strategy:
  - Tier 1 (zero API cost): retrieval-level stability for (1) and (2).
    Re-implements Eq. 2 exactly; shows the context set barely changes.
  - Tier 2 (~cents in embedding API cost, cached): context-volume and
    case-level analysis for (3) using the real text-embedding-3-small
    cosines — the same metric space the live system operates in.
    No GPT-4o regeneration required for the volume sweep.

Requires OPENAI_API_KEY env var for Tier 2.

Usage:
  python sensitivity_analysis_e11.py
"""

import hashlib
import json
import os
import re
import sys
import asyncio
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import openai as _openai_lib
from visit_ids import build_visit_id_map, extract_audio_key

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(".")
TRANS_DIR  = BASE_DIR / "Transcripts_Deepgram_nova-2_order by patient"
PROFILES_CSV   = BASE_DIR / "patient_profiles copy.csv"
ACRAG_CSV      = BASE_DIR / "visiting_forms_ground_truth_ACRAG_deepgram_v1.csv"
OUTPUT_DIR     = BASE_DIR / "sensitivity_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Default heuristic values (paper's operating point)
DEFAULT_W_AGE    = 0.7
DEFAULT_AGE_BINS = [5, 10, 15]        # ≤5 → 1.0, ≤10 → 0.8, ≤15 → 0.6, else → 0.4
DEFAULT_TAU_HYBRID  = 0.75
DEFAULT_TAU_CROSS   = 0.55

# Sweep grids
W_AGE_GRID       = [0.5, 0.6, 0.7, 0.8, 0.9]    # w_gender = 1 - w_age
AGE_BINS_GRID    = [                              # (tight, default, loose)
    [3,  6, 10],   # tight
    [5, 10, 15],   # default ← paper value
    [7, 12, 17],   # loose
]
TAU_GRID         = [0.45, 0.55, 0.65, 0.75, 0.85]  # both τ swept simultaneously
GENDER_MISMATCH  = 0.3   # fixed (match→1.0, mismatch→0.3)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_patient_profiles() -> pd.DataFrame:
    df = pd.read_csv(PROFILES_CSV)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["disease"]    = df["disease"].astype(str).str.lower().str.strip()
    df["gender"]     = df["gender"].astype(str).str.strip().str.upper()
    df["age"]        = pd.to_numeric(df["age"], errors="coerce")
    return df.set_index("patient_id")


def build_disease_groups(profiles: pd.DataFrame) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for pid, row in profiles.iterrows():
        d = row["disease"]
        groups.setdefault(d, []).append(pid)
    return groups


def load_transcripts() -> Dict[str, str]:
    """Return dict: patient_id → concatenated transcript text (all visits)"""
    texts: Dict[str, str] = {}
    if not TRANS_DIR.exists():
        return texts
    for txt_file in TRANS_DIR.rglob("*.txt"):
        pid = txt_file.parent.name
        texts.setdefault(pid, "")
        texts[pid] += txt_file.read_text(encoding="utf-8", errors="ignore") + "\n"
    return texts


def load_eval_cases() -> pd.DataFrame:
    df = pd.read_csv(ACRAG_CSV)
    df["patient_id"] = df["File Name"].str.split("/").str[0]
    visit_keys = df["File Name"].map(extract_audio_key)
    if visit_keys.isna().any() or visit_keys.duplicated().any():
        raise ValueError("Sensitivity input must contain unique visit audio keys")
    visit_ids = build_visit_id_map(visit_keys)
    df["visit_id"] = visit_keys.map(visit_ids)
    return df


# ── Demographic similarity (implements Appendix Eq. 2) ───────────────────────

def age_sim_score(age_a: float, age_b: float, bins: List[int]) -> float:
    """
    Tiered age similarity.
    bins = [b0, b1, b2] → diff≤b0→1.0, ≤b1→0.8, ≤b2→0.6, else→0.4
    Missing values → 0.5 (neutral).
    """
    if np.isnan(age_a) or np.isnan(age_b):
        return 0.5
    diff = abs(age_a - age_b)
    if diff <= bins[0]:   return 1.0
    if diff <= bins[1]:   return 0.8
    if diff <= bins[2]:   return 0.6
    return 0.4


def gender_sim_score(g_a: str, g_b: str) -> float:
    if not g_a or not g_b or g_a == "NAN" or g_b == "NAN":
        return 0.5
    return 1.0 if g_a == g_b else GENDER_MISMATCH


def demographic_score(pid_a: str, pid_b: str,
                      profiles: pd.DataFrame,
                      w_age: float, age_bins: List[int]) -> float:
    """
    score = age_sim × w_age + gender_sim × (1 − w_age)
    """
    if pid_a not in profiles.index or pid_b not in profiles.index:
        return 0.5
    a, b = profiles.loc[pid_a], profiles.loc[pid_b]
    a_sim = age_sim_score(a["age"], b["age"], age_bins)
    g_sim = gender_sim_score(a["gender"], b["gender"])
    return a_sim * w_age + g_sim * (1 - w_age)


def get_top_k(target_pid: str,
              profiles: pd.DataFrame,
              disease_groups: Dict[str, List[str]],
              k: int,
              w_age: float,
              age_bins: List[int]) -> List[str]:
    """
    Return top-k same-disease patient IDs for target_pid using given heuristics.
    """
    if target_pid not in profiles.index:
        return []
    disease = profiles.loc[target_pid, "disease"]
    candidates = [p for p in disease_groups.get(disease, []) if p != target_pid]
    if not candidates:
        return []
    scored = [(p, demographic_score(target_pid, p, profiles, w_age, age_bins))
              for p in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored[:k]]


# ── Retrieval stability metrics ───────────────────────────────────────────────

def kendall_rank_corr(list_a: List[str], list_b: List[str]) -> float:
    """
    Kendall's τ between two ranked lists of patient IDs.
    Use the union as domain; missing items scored at position len+1.
    Returns NaN if either list is empty.
    """
    union = list(dict.fromkeys(list_a + list_b))  # ordered unique
    if len(union) < 2:
        return float("nan")

    def to_rank_vec(lst):
        rank = {p: i for i, p in enumerate(lst)}
        return [rank.get(p, len(lst)) for p in union]

    τ, _ = kendalltau(to_rank_vec(list_a), to_rank_vec(list_b))
    return float(τ)


def jaccard_at_k(set_a: List[str], set_b: List[str]) -> float:
    sa, sb = set(set_a), set(set_b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


# ── Tier 1: Demographic weight sweep ─────────────────────────────────────────

def tier1_weight_sweep(profiles: pd.DataFrame,
                       disease_groups: Dict[str, List[str]],
                       k: int = 3) -> pd.DataFrame:
    """
    Sweep w_age; compare top-k sets against default setting.
    """
    print("\n── Tier 1A: Demographic weight sweep ──────────────────────────────")
    eval_patients = [pid for pid in profiles.index
                     if any(pid in g for g in disease_groups.values()
                            if len(g) > 1)]

    # Default context set per patient
    default_sets = {
        pid: get_top_k(pid, profiles, disease_groups, k,
                       DEFAULT_W_AGE, DEFAULT_AGE_BINS)
        for pid in eval_patients
    }

    rows = []
    for w_age in W_AGE_GRID:
        taus, jaccs = [], []
        changed_patients = 0
        for pid in eval_patients:
            variant = get_top_k(pid, profiles, disease_groups, k, w_age, DEFAULT_AGE_BINS)
            default = default_sets[pid]
            if not default and not variant:
                continue
            τ  = kendall_rank_corr(default, variant)
            j  = jaccard_at_k(default, variant)
            if not np.isnan(τ): taus.append(τ)
            jaccs.append(j)
            if set(variant) != set(default):
                changed_patients += 1

        is_default = "← default" if abs(w_age - DEFAULT_W_AGE) < 1e-9 else ""
        rows.append({
            "w_age":   w_age,
            "w_gender": round(1 - w_age, 1),
            "mean_kendall_tau":   round(np.nanmean(taus),  4) if taus  else "—",
            "mean_jaccard_top3":  round(np.mean(jaccs), 4)    if jaccs else "—",
            "patients_context_changed": changed_patients,
            "note": is_default,
        })
        print(f"  w_age={w_age:.1f}  τ={rows[-1]['mean_kendall_tau']:.4f}  "
              f"J={rows[-1]['mean_jaccard_top3']:.4f}  "
              f"changed={changed_patients}/{len(eval_patients)} {is_default}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "sensitivity_tier1_weights.csv", index=False)
    print(f"  Saved → sensitivity_results/sensitivity_tier1_weights.csv")
    return df


# ── Tier 1: Age threshold sweep ───────────────────────────────────────────────

def tier1_age_threshold_sweep(profiles: pd.DataFrame,
                               disease_groups: Dict[str, List[str]],
                               k: int = 3) -> pd.DataFrame:
    """
    Sweep age-similarity bin cutoffs; compare context sets against default.
    """
    print("\n── Tier 1B: Age-similarity threshold sweep ──────────────────────")
    eval_patients = [pid for pid in profiles.index
                     if any(pid in g for g in disease_groups.values()
                            if len(g) > 1)]

    default_sets = {
        pid: get_top_k(pid, profiles, disease_groups, k,
                       DEFAULT_W_AGE, DEFAULT_AGE_BINS)
        for pid in eval_patients
    }

    bin_labels = ["tight [3,6,10]", "default [5,10,15]", "loose [7,12,17]"]
    rows = []
    for bins, label in zip(AGE_BINS_GRID, bin_labels):
        taus, jaccs = [], []
        changed_patients = 0
        for pid in eval_patients:
            variant = get_top_k(pid, profiles, disease_groups, k, DEFAULT_W_AGE, bins)
            default = default_sets[pid]
            if not default and not variant:
                continue
            τ = kendall_rank_corr(default, variant)
            j = jaccard_at_k(default, variant)
            if not np.isnan(τ): taus.append(τ)
            jaccs.append(j)
            if set(variant) != set(default):
                changed_patients += 1

        is_default = "← default" if bins == DEFAULT_AGE_BINS else ""
        rows.append({
            "age_bins":  str(bins),
            "label":     label,
            "mean_kendall_tau":   round(np.nanmean(taus),  4) if taus  else "—",
            "mean_jaccard_top3":  round(np.mean(jaccs), 4)    if jaccs else "—",
            "patients_context_changed": changed_patients,
            "note": is_default,
        })
        print(f"  {label}  τ={rows[-1]['mean_kendall_tau']:.4f}  "
              f"J={rows[-1]['mean_jaccard_top3']:.4f}  "
              f"changed={changed_patients}/{len(eval_patients)} {is_default}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "sensitivity_tier1_age_thresholds.csv", index=False)
    print(f"  Saved → sensitivity_results/sensitivity_tier1_age_thresholds.csv")
    return df


# ── Embedding utilities (real text-embedding-3-small) ────────────────────────

EMBEDDING_CACHE_FILE = OUTPUT_DIR / "embedding_cache.json"
EMBEDDING_MODEL      = "text-embedding-3-small"
EMBEDDING_DIM        = 1536
EMBED_CONCURRENCY    = 8   # parallel embedding requests


def load_embedding_cache() -> Dict[str, List[float]]:
    if EMBEDDING_CACHE_FILE.exists():
        with open(EMBEDDING_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_embedding_cache(cache: Dict[str, List[float]]) -> None:
    with open(EMBEDDING_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_text(text: str, max_tokens: int = 8000) -> List[str]:
    """
    Replicates chunk_text_for_embedding() from the live notebook.
    Splits on Japanese / English sentence boundaries; uses tiktoken for
    token counting when available, falls back to character estimation.
    For typical home-nursing visit transcripts (<= ~4000 tokens) this
    returns a single chunk — identical to a plain single-call embedding.
    """
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model("text-embedding-3-small")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return [text]
        # Split by sentence boundaries (Japanese first, then English/lines)
        sentences = text.split("。")
        if len(sentences) == 1:
            sentences = text.split(".")
        if len(sentences) == 1:
            sentences = text.split("\n")
        chunks, current, current_tok = [], "", 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            st = len(enc.encode(s + "。"))
            if st > max_tokens:                        # single sentence too long
                chunks.append(s[:max_tokens * 3])     # rough char fallback
                continue
            if current_tok + st > max_tokens and current:
                chunks.append(current.strip())
                current, current_tok = s + "。", st
            else:
                current += s + "。"
                current_tok += st
        if current.strip():
            chunks.append(current.strip())
        return chunks or [text]
    except ImportError:
        # tiktoken unavailable: nursing transcripts are typically short enough
        # that 8000-char truncation covers the whole text (safe fallback).
        return [text[:24000]]   # 24000 chars ≈ 8000 tokens for Japanese


async def get_embedding(
    text: str,
    client: _openai_lib.AsyncOpenAI,
    cache: Dict[str, List[float]],
    semaphore: asyncio.Semaphore,
) -> np.ndarray:
    """
    Replicates the live get_averaged_embedding() behaviour:
      - chunk at sentence boundaries up to 8000 tokens (via tiktoken)
      - single chunk  → one API call  (the common case for nursing visits)
      - multiple chunks → gather + average (same as live system)
    Cache key is the SHA-256 digest of the full original text so chunking is transparent.
    """
    text = text.strip()
    if not text:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    key = text_key(text)
    if key in cache:
        return np.array(cache[key], dtype=np.float32)

    chunks = _chunk_text(text)

    async def _embed_one(chunk: str) -> np.ndarray:
        async with semaphore:
            try:
                resp = await client.embeddings.create(
                    model=EMBEDDING_MODEL, input=chunk.strip()
                )
                return np.array(resp.data[0].embedding, dtype=np.float32)
            except Exception as e:
                print(f"  [embed error] {e}")
                return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    if len(chunks) == 1:
        emb = await _embed_one(chunks[0])
    else:
        embs  = await asyncio.gather(*[_embed_one(c) for c in chunks])
        valid = [e for e in embs if not np.allclose(e, 0)]
        emb   = (np.mean(valid, axis=0).astype(np.float32) if valid
                 else np.zeros(EMBEDDING_DIM, dtype=np.float32))

    cache[key] = emb.tolist()
    return emb


def emb_cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def load_visit_transcripts() -> Dict[str, List[str]]:
    """
    Return dict: patient_id → list of individual visit transcript texts.
    Mirrors how the live system stores per-record contexts.
    """
    visits: Dict[str, List[str]] = {}
    if not TRANS_DIR.exists():
        return visits
    for txt_file in sorted(TRANS_DIR.rglob("*.txt")):
        pid = txt_file.parent.name
        visits.setdefault(pid, [])
        text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            visits[pid].append(text)
    return visits


async def prefetch_embeddings(
    texts: List[str],
    client: _openai_lib.AsyncOpenAI,
    cache: Dict[str, List[float]],
    label: str = "texts",
) -> None:
    """Embed a list of texts in parallel, flushing cache periodically."""
    sem = asyncio.Semaphore(EMBED_CONCURRENCY)
    uncached = [t for t in texts
                if text_key(t[:8000].strip()) not in cache and t.strip()]
    print(f"  Prefetching {label}: {len(uncached)} new / {len(texts)} total")
    if not uncached:
        return
    tasks = [get_embedding(t, client, cache, sem) for t in uncached]
    for i, result in enumerate(asyncio.as_completed(tasks), 1):
        await result
        if i % 20 == 0 or i == len(uncached):
            save_embedding_cache(cache)
            print(f"    [{i}/{len(uncached)}] cached")
    save_embedding_cache(cache)


# ── Tier 2A: Context volume sweep — real embedding cosines ────────────────────

async def tier2_volume_sweep_embedding(
    profiles: pd.DataFrame,
    disease_groups: Dict[str, List[str]],
    visit_transcripts: Dict[str, List[str]],
    eval_cases: pd.DataFrame,
    client: _openai_lib.AsyncOpenAI,
    cache: Dict[str, List[float]],
) -> pd.DataFrame:
    """
    For each τ, count how many (current-visit-transcript, candidate-record)
    pairs exceed the embedding cosine threshold — exactly as
    calculate_context_relevance() does in the live system.
    """
    print("\n── Tier 2A: Context volume sweep (real text-embedding-3-small) ───")
    sem = asyncio.Semaphore(EMBED_CONCURRENCY)

    # Collect every unique text that will be needed, then prefetch
    all_pids: set = set(eval_cases["patient_id"].tolist())
    for pid in list(all_pids):
        if pid in profiles.index:
            disease = profiles.loc[pid, "disease"]
            for cand in disease_groups.get(disease, []):
                all_pids.add(cand)

    all_texts: List[str] = []
    for pid in all_pids:
        all_texts.extend(visit_transcripts.get(pid, []))
    await prefetch_embeddings(
        list(dict.fromkeys(all_texts)), client, cache,
        label="all patient transcripts/records"
    )

    rows = []
    for tau in TAU_GRID:
        n_above_list, pct_nonzero_list = [], []

        for _, row in eval_cases.iterrows():
            pid = row["patient_id"]
            if pid not in profiles.index:
                continue
            curr_texts = visit_transcripts.get(pid, [])
            if not curr_texts:
                continue

            # Stage 1: top-3 demographically similar same-disease patients
            # — identical to find_same_disease_patients_only() in the live system.
            candidates = get_top_k(pid, profiles, disease_groups,
                                   k=3, w_age=DEFAULT_W_AGE, age_bins=DEFAULT_AGE_BINS)

            n_above = 0
            for curr_text in curr_texts:
                curr_emb = await get_embedding(curr_text, client, cache, sem)
                for cand in candidates:
                    for cand_record in visit_transcripts.get(cand, []):
                        cand_emb = await get_embedding(cand_record, client, cache, sem)
                        if emb_cosine(curr_emb, cand_emb) >= tau:
                            n_above += 1

            n_above_list.append(n_above)
            pct_nonzero_list.append(1 if n_above > 0 else 0)

        hybrid_label = "← hybrid default"    if abs(tau - DEFAULT_TAU_HYBRID) < 1e-9 else ""
        cross_label  = "← cross-only default" if abs(tau - DEFAULT_TAU_CROSS)  < 1e-9 else ""
        note = hybrid_label or cross_label

        rows.append({
            "tau":                    tau,
            "mean_contexts_passing":  round(np.mean(n_above_list),    3) if n_above_list    else "—",
            "pct_cases_with_context": round(np.mean(pct_nonzero_list), 3) if pct_nonzero_list else "—",
            "note": note,
        })
        print(f"  τ={tau:.2f}  mean_contexts={rows[-1]['mean_contexts_passing']}  "
              f"pct_with_context={rows[-1]['pct_cases_with_context']}  {note}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "sensitivity_tier2_volume_embedding.csv", index=False)
    print(f"  Saved → sensitivity_results/sensitivity_tier2_volume_embedding.csv")
    return df


# ── Tier 2B: Targeted case study — real embedding cosines ─────────────────────

async def tier2_case_study_embedding(
    profiles: pd.DataFrame,
    disease_groups: Dict[str, List[str]],
    visit_transcripts: Dict[str, List[str]],
    eval_cases: pd.DataFrame,
    client: _openai_lib.AsyncOpenAI,
    cache: Dict[str, List[float]],
) -> pd.DataFrame:
    """
    For each eval case, compute the max embedding cosine to any same-disease
    candidate record and assign to a τ bucket.  Main-eval ROUGE-L is looked up
    from the existing AC-RAG output CSV so the metric scale is consistent with
    Table 4 in the manuscript — no re-generation needed.
    Cases whose context set changes between τ=0.55 and τ=0.75 are flagged.
    """
    print("\n── Tier 2B: Case study — real embedding cosines ──────────────────")
    sem = asyncio.Semaphore(EMBED_CONCURRENCY)

    def rouge_l_char(ref: str, hyp: str) -> float:
        a = re.sub(r'\s+', '', str(ref))
        b = re.sub(r'\s+', '', str(hyp))
        m, n = len(a), len(b)
        if not m or not n:
            return 0.0
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                curr[j] = (prev[j - 1] + 1 if a[i - 1] == b[j - 1]
                           else max(curr[j - 1], prev[j]))
            prev = curr
        lcs = prev[n]
        p = lcs / n; r = lcs / m
        return 2 * p * r / (p + r) if p + r else 0.0

    rows = []
    for _, row in eval_cases.iterrows():
        pid = row["patient_id"]
        if pid not in profiles.index:
            continue
        curr_texts = visit_transcripts.get(pid, [])
        if not curr_texts:
            continue

        # Stage 1: top-3 demographically similar same-disease patients
        # — identical to find_same_disease_patients_only() in the live system.
        candidates = get_top_k(pid, profiles, disease_groups,
                               k=3, w_age=DEFAULT_W_AGE, age_bins=DEFAULT_AGE_BINS)

        max_sim = 0.0
        for curr_text in curr_texts:
            curr_emb = await get_embedding(curr_text, client, cache, sem)
            for cand in candidates:
                for cand_record in visit_transcripts.get(cand, []):
                    cand_emb = await get_embedding(cand_record, client, cache, sem)
                    sim = emb_cosine(curr_emb, cand_emb)
                    if sim > max_sim:
                        max_sim = sim

        if   max_sim >= 0.85: bucket = "≥0.85"
        elif max_sim >= 0.75: bucket = "0.75–0.85"
        elif max_sim >= 0.65: bucket = "0.65–0.75"
        elif max_sim >= 0.55: bucket = "0.55–0.65"
        elif max_sim >= 0.45: bucket = "0.45–0.55"
        else:                  bucket = "<0.45 (no context at any τ)"

        gt    = str(row.get("Ground Truth", ""))
        acrag = str(row.get("LLM based summarization", ""))
        rl    = rouge_l_char(gt, acrag)

        rows.append({
            "visit_id":                             row["visit_id"],
            "max_embedding_cosine_same_disease":    round(max_sim, 4),
            "context_threshold_bucket":             bucket,
            "main_eval_rouge_l":                    round(rl, 4),
            "gets_context_at_tau_0.55":             max_sim >= 0.55,
            "gets_context_at_tau_0.75":             max_sim >= 0.75,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "sensitivity_tier2_case_study_embedding.csv", index=False)

    print("\n  Distribution of cases by max embedding cosine to same-disease candidate:")
    for bucket, grp in df.groupby("context_threshold_bucket", sort=False):
        print(f"    {bucket:32s}  n={len(grp):3d}  "
              f"mean ROUGE-L = {grp['main_eval_rouge_l'].mean():.4f}")

    n55  = int(df["gets_context_at_tau_0.55"].sum())
    n75  = int(df["gets_context_at_tau_0.75"].sum())
    ntot = len(df)
    print(f"\n  Cases receiving context at operational τ=0.55 (cross-only): "
          f"{n55}/{ntot} ({100*n55/ntot:.1f}%)")
    print(f"  Cases receiving context at τ=0.75 (hybrid):            "
          f"{n75}/{ntot} ({100*n75/ntot:.1f}%)")
    print(f"  Cases whose context set changes between τ=0.55 and 0.75: "
          f"{n55 - n75}/{ntot} ({100*(n55 - n75)/ntot:.1f}%)")
    print(f"  Cases with no context at any τ: "
          f"{(df['max_embedding_cosine_same_disease'] < 0.45).sum()}/{ntot}")
    print(f"\n  Saved → sensitivity_results/sensitivity_tier2_case_study_embedding.csv")
    return df


# ── (removed) TF-IDF proxy functions ─────────────────────────────────────────
# compute_tfidf_cosine, tier2_context_volume_sweep, tier2_targeted_case_study,
# build_prompt_for_tau, compute_rouge_l, _process_one_case,
# tier2_generation_sweep_async, tier2_generation_sweep
# were all removed. Tier 2 now uses real text-embedding-3-small cosines.
# See git history if the TF-IDF proxy is needed for reference.
#
# Rationale: τ does not transfer across metric spaces.  A cutoff of 0.55 in
# embedding cosine space and 0.55 in sparse char-TF-IDF space select completely
# different fractions of pairs.  The proxy sweep was answering the wrong
# question.  The embedding volume sweep (above) answers E11 in the space the
# system actually operates in.

# Placeholder for removed function
def compute_tfidf_cosine(text_a: str, text_b: str) -> float:  # noqa: F841
    """Removed — use real embedding cosines instead."""
    raise NotImplementedError(
        "compute_tfidf_cosine was removed. "
        "Use tier2_volume_sweep_embedding / tier2_case_study_embedding."
    )


# ── CONCURRENCY constant (kept here so nothing else breaks if referenced) ─────
CONCURRENCY = EMBED_CONCURRENCY  # alias


# ── dummy reference so old function
def tier2_context_volume_sweep(*a, **kw):  # noqa: F841
    raise NotImplementedError("Replaced by tier2_volume_sweep_embedding.")


def tier2_targeted_case_study(*a, **kw):  # noqa: F841
    raise NotImplementedError("Replaced by tier2_case_study_embedding.")


def tier2_generation_sweep(*a, **kw):  # noqa: F841
    raise NotImplementedError("Generation sweep removed — see module docstring.")


# ── Summary report ────────────────────────────────────────────────────────────

def print_summary(df_w, df_age, df_vol, df_cases):
    sep = "=" * 65
    print(f"\n{sep}")
    print("SENSITIVITY ANALYSIS SUMMARY — E11")
    print(sep)

    print("\nTable 1: Demographic weight sweep (Equation 2)")
    print(df_w[["w_age", "w_gender", "mean_kendall_tau",
                "mean_jaccard_top3", "patients_context_changed", "note"]].to_string(index=False))

    print("\nTable 2: Age-similarity threshold sweep")
    print(df_age[["label", "mean_kendall_tau",
                  "mean_jaccard_top3", "patients_context_changed", "note"]].to_string(index=False))

    print("\nTable 3: Semantic threshold — context volume (real text-embedding-3-small)")
    print(df_vol[["tau", "mean_contexts_passing",
                  "pct_cases_with_context", "note"]].to_string(index=False))

    print("\nTable 4: Cases by max embedding cosine to same-disease candidate")
    print("  (main-eval ROUGE-L = GT vs full AC-RAG output, same scale as manuscript Table 4)")
    by_bucket = df_cases.groupby("context_threshold_bucket").agg(
        n=("main_eval_rouge_l", "count"),
        mean_rouge_l=("main_eval_rouge_l", "mean"),
        std_rouge_l=("main_eval_rouge_l", "std"),
    ).reset_index()
    print(by_bucket.to_string(index=False))

    sep2 = "-" * 65
    print(f"\n{sep2}")
    print("MANUSCRIPT FILL-IN PLACEHOLDERS")
    print(sep2)

    j_vals = pd.to_numeric(df_w["mean_jaccard_top3"], errors="coerce")
    k_vals = pd.to_numeric(df_w["mean_kendall_tau"],  errors="coerce")
    print(f"  [JACCARD_W_RANGE]  {j_vals.min():.3f}–{j_vals.max():.3f}")
    print(f"  [KENDALL_W_RANGE]  {k_vals.min():.3f}–{k_vals.max():.3f}")

    n55  = int(df_cases["gets_context_at_tau_0.55"].sum())
    n75  = int(df_cases["gets_context_at_tau_0.75"].sum())
    ntot = len(df_cases)
    print(f"  [N_AFFECTED_55]    {n55}/{ntot} cases ({100*n55/ntot:.1f}%) receive context at τ=0.55")
    print(f"  [N_AFFECTED_75]    {n75}/{ntot} cases ({100*n75/ntot:.1f}%) receive context at τ=0.75")
    print(f"  [N_CHANGED_55_75]  {n55 - n75}/{ntot} cases change context set between τ=0.55 and 0.75")

    rl_all = df_cases["main_eval_rouge_l"]
    print(f"  [ROUGE_L_OVERALL]  {rl_all.mean():.4f} ± {rl_all.std():.4f}  "
          f"(all cases, main AC-RAG eval)")

    affected = df_cases[df_cases["gets_context_at_tau_0.55"]]
    if not affected.empty:
        print(f"  [ROUGE_L_AFFECTED] {affected['main_eval_rouge_l'].mean():.4f} ± "
              f"{affected['main_eval_rouge_l'].std():.4f}  "
              f"(the {len(affected)} cases receiving context at operational τ=0.55)")

    print(f"\n{sep}")
    print("All CSVs saved to: sensitivity_results/")
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

async def async_main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("       Tier 1 (demographic sweep) requires no API key and can be")
        print("       extracted here manually, but Tier 2 needs it.")
        sys.exit(1)

    print("Loading data...")
    profiles          = load_patient_profiles()
    disease_groups    = build_disease_groups(profiles)
    visit_transcripts = load_visit_transcripts()
    eval_cases        = load_eval_cases()
    print(f"  Profiles: {len(profiles)} patients  "
          f"Disease groups: {len(disease_groups)}  "
          f"Patients with transcripts: {len(visit_transcripts)}  "
          f"Eval cases: {len(eval_cases)}")

    # ── Tier 1: demographic sweep — zero API cost ─────────────────────────────
    df_weights = tier1_weight_sweep(profiles, disease_groups)
    df_age     = tier1_age_threshold_sweep(profiles, disease_groups)

    # ── Tier 2: real text-embedding-3-small cosine sweep ─────────────────────
    client = _openai_lib.AsyncOpenAI(api_key=api_key, timeout=60.0, max_retries=3)
    cache  = load_embedding_cache()

    df_volume = await tier2_volume_sweep_embedding(
        profiles, disease_groups, visit_transcripts, eval_cases, client, cache
    )
    df_cases = await tier2_case_study_embedding(
        profiles, disease_groups, visit_transcripts, eval_cases, client, cache
    )

    print_summary(df_weights, df_age, df_volume, df_cases)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
