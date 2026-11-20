# Adaptive Cascaded Retrieval-Augmented Generation (AC-RAG)

Extraction and Summarization for Home-Visit Nursing Records.

Research notebooks for the transcript-to-documentation pipeline (information extraction, AC-RAG summarization, and baselines).

## Code availability

The source code supporting this work is available at [GitHub](https://github.com/milyun/adaptive-cascaded-retrieval-augmented-generation-ac-rag). The archived v0.1.0 release is available on Zenodo at [doi:10.5281/zenodo.23656597](https://doi.org/10.5281/zenodo.23656597). Release v0.2.0 is prepared for publication; its version-specific Zenodo DOI will be added after the archive is published. The Zenodo concept DOI is [10.5281/zenodo.23656596](https://doi.org/10.5281/zenodo.23656596). The repository includes the pipeline and prompt notebooks, evaluation modules, configuration, and synthetic example records; clinical data are not included.

The clinical data (audio, transcripts, nurse-written records, patient profiles) contain personal health information and cannot be released. See the Data Availability statement in the manuscript for the access procedure. `sample_data/` contains synthetic example records that document the expected file formats.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...   # never commit keys
```

Data paths in the notebooks default to `data/`. Place your own data there, following the layout in `sample_data/`.

## Layout

| Component | Files |
|---|---|
| Pipelines, prompts, and baselines | `enhanced-RAG_extractionSummarization.ipynb`, `enhanced-RAG_extractionSummarization_v1.ipynb`, `original-RAG_extractionSummarization.ipynb`, `original-RAG_baseline.ipynb`, `multi-stage_extractionSummarization.ipynb`, `singlePipeline_extractionSummarization.ipynb`, `zero-shot_baseline.ipynb`, `fs-gkp (original).ipynb` |
| Evaluation | `clinical_metrics.py`, `evaluation_pipeline.py`, `evaluation_pipeline_multi.py` |
| Configuration | `requirements.txt` |
| Synthetic input examples | `sample_data/` |

## Reproducibility

The reported AC-RAG results were produced by `enhanced-RAG_extractionSummarization_v1.ipynb`. The Whisper usage reported in Table 6 was re-measured on 8 November 2026. The Deepgram ProcessingMeter output is from 7 January 2026. That run logged aggregate totals only; the GPT-4o/embedding token split is derived from the logged embedding cost, not separately recorded token counts. The scripts in `reproducibility/` read local evaluation inputs; clinical source data are not included.

FS-GKP extraction token counts were counted with `tiktoken` from input and output texts, excluding the prompt template; extraction costs in the source logs use the earlier $5/$15 per-million input/output prices. The stage-2 `summary_time_cost` files also contain aggregate-only totals rather than per-visit rows. Accordingly, the release preserves the original aggregate logs and does not infer or distribute stage-2 token, cost, or time totals among visits. Run `python reproducibility/reprice_fsgkp_table6.py` after exporting the logs to reproduce the combined-stage Table 6 costs at $2.50/$10 per million tokens.

For a full evaluation, run `batch_evaluate_all_systems.py --source-dir <local-data-dir>` and then `export_deidentified_outputs.py --source-dir <local-data-dir> --output-dir <reproducibility-dir>`. The first step writes local evaluation intermediates; only the de-identified exports belong in a public release.

| Item | Reproduction code | Output |
|---|---|---|
| Table 2 | `reproducibility/table2_scoring.py`, `reproducibility/compute_extraction_prf1_table2.py` | `table2_field_results.csv`, `table2_precision_recall_f1.csv` |
| Table 2 constant-mode baseline | `reproducibility/compute_constant_mode_baseline_table2.py` | `constant_mode_baseline_table2.csv`, `constant_mode_baseline_table2.md` |
| Table 3 | `reproducibility/batch_evaluate_all_systems.py`, `reproducibility/compute_tables3_5_statistics.py` | `detailed_per_sample_results.csv`, `table3_system_summary.csv` |
| Table 3 effect sizes | `reproducibility/compute_tables3_5_statistics.py` | `table3_pairwise_effect_sizes.csv` |
| Table 4 Friedman/Wilcoxon tests | `reproducibility/compute_tables3_5_statistics.py` | `table4_friedman_wilcoxon.csv` |
| Table 4 nurse ratings | `reproducibility/compute_nurse_rating_statistics.py` | `table4_nurse_rating_summary.csv`, `table4_nurse_rating_tests.csv` |
| Table 5 | `reproducibility/compute_tables3_5_statistics.py` | `table5_bertscore_summary.csv`, `table5_bertscore_mann_whitney.csv` |
| Table 6 | `reproducibility/export_deidentified_outputs.py`, `reproducibility/reprice_fsgkp_table6.py` | `table6_usage_summary.csv`, `processing_meter_deepgram.json`, `api_call_log_whisper.csv`, `processing_meter_whisper.json`, `table6_logs/acrag_deepgram_meter_20260107.txt`, `table6_logs/fsgkp_stage1_extraction_*.csv`, `table6_logs/fsgkp_stage2_summary_*.txt`, `table6_logs/fsgkp_stage2_summary_totals.csv`, `table6_logs/fsgkp_repriced_costs.csv` |
| Figure 2 | `reproducibility/generate_jmir_figures.py` | `jmir_figures/Figure2_Method_Comparison.png` and `.pdf` |
| Figure 3 | `reproducibility/generate_jmir_figures.py` | `jmir_figures/Figure3_Performance_Heatmap.png` and `.pdf` |
| Figure 4 | `reproducibility/generate_jmir_figures.py` | `jmir_figures/Figure4_ASR_Comparison.png` and `.pdf` |
| Figure 5 | `reproducibility/generate_jmir_figures.py` | `jmir_figures/Figure5_Radar_Chart_Top4.png` and `.pdf` |
| Figure 6 | `reproducibility/generate_nurse_rating_figure.py` | `figures/Figure6_nurse_ratings.png` and `.pdf` |
| Multimedia Appendix 3 sensitivity analysis | `reproducibility/sensitivity_analysis_e11.py`, `reproducibility/export_sensitivity_summaries.py` | Aggregate CSVs in `reproducibility/sensitivity_results/` |
| Multimedia Appendix 4 entity-recall audit | `reproducibility/export_entity_dictionary.py`, `reproducibility/export_entity_recall_audit.py` | `reproducibility/entity_recall_audit/entity_dictionary.csv`, `entity_recall_term_list.csv` |

## Suggested Citation

Shoumi MN, Harada K, Oshita H, Sakashita S, Inoue S. Adaptive Cascaded Retrieval-Augmented Generation (AC-RAG): source code and reproducibility materials, v0.2.0. Zenodo. 2026. Concept DOI: [10.5281/zenodo.23656596](https://doi.org/10.5281/zenodo.23656596). Version-specific DOI pending publication.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
