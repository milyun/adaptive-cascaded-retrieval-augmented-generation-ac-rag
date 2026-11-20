#!/usr/bin/env python3
"""
Clinical Summarization Evaluation Metrics
Additional metrics specifically for medical documentation
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
from janome.tokenizer import Tokenizer
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# NLP evaluation metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    warnings.warn("rouge-score not installed. Install with: pip install rouge-score")

try:
    import sacrebleu
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    warnings.warn("sacrebleu not installed. Install with: pip install sacrebleu")

def check_bleu_available():
    """Dynamically check if sacrebleu is available (handles post-import installation)"""
    try:
        import sacrebleu
        return True
    except ImportError:
        return False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    warnings.warn("bert-score not installed. Install with: pip install bert-score")

def check_bertscore_available():
    """Dynamically check if bert-score is available (handles post-import installation)"""
    try:
        from bert_score import score as bert_score
        return True
    except ImportError:
        return False

class ClinicalMetrics:
    """Clinical-specific evaluation metrics"""

    def __init__(self):
        self.tokenizer = Tokenizer()

        # Define medical keywords for entity extraction
        self.entity_patterns = {
            'medications': [
                'ニトロ', 'コーラック', 'リンゼス', 'モビコール', 'ベタヒスメチル',
                'ニュープロパッチ', '内服', '服薬', '薬', 'mg', 'ml', '錠', 'アリセプト',
                'メマリー', 'レミニール', 'ラシックス', 'ノルバスク', 'プレドニン', 'リリカ',
                'カロナール', 'ロキソニン'
            ],
            'vital_signs': [
                '血圧', '脈拍', '体温', '酸素', 'SpO2', 'バイタル'
            ],
            'symptoms': [
                '痛み', '痺れ', '発熱', '転倒', '嘔吐', '下痢', '便秘', '排便',
                '耳鳴り', '頭痛', '掻痒感', 'かゆみ', '痰', '息苦しい', '動悸'
            ],
            'body_parts': [
                '右足', '左足', '上肢', '下肢', '腹部', '胸部', '頭部', '膝'
            ],
            'procedures': [
                '入浴', '更衣', '処置', '訓練', 'マッサージ', '観察', 'リハビリ',
                '食事介助', '排泄介助', '移乗', '体位変換', '清拭', '口腔ケア', '爪切り',
                '整容', '歩行介助', '見守り', '声かけ'
            ],
            'diseases': [
                '認知症', 'パーキンソン病', '糖尿病', '高血圧', '心不全', '脳梗塞', '褥瘡', '誤嚥', '肺炎', '骨折', '浮腫', '脱水'
            ]
        }

        # Required sections in visiting nurse forms
        self.required_sections = {
            '利用者の状態': r'【?利用者の状態',
            'トピックス': r'トピックス|【トピックス】',
            '処置内容': r'処置内容|<処置内容',
            '訓練内容': r'訓練内容',
            '備考': r'備考|<備考',
            '外部連絡': r'外部連絡|<外部連絡',
            '内部連絡': r'内部連絡|<内部連絡'
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract clinical entities from text"""
        entities = {category: [] for category in self.entity_patterns.keys()}

        for category, keywords in self.entity_patterns.items():
            for keyword in keywords:
                # Find keyword with some context
                pattern = rf'(.{{0,20}}{keyword}.{{0,20}})'
                matches = re.findall(pattern, text)
                entities[category].extend(matches)

        return entities

    def calculate_entity_recall(self, ground_truth: str, generated: str) -> Dict[str, float]:
        """
        Calculate recall of clinical entities based on keyword presence
        Rather than matching extracted phrases, we check if keywords appear in the text
        """
        recalls = {}

        # Normalize texts for comparison
        gt_normalized = self._normalize_text(ground_truth)
        gen_normalized = self._normalize_text(generated)

        for category, keywords in self.entity_patterns.items():
            # Count how many keywords from this category appear in each text
            gt_count = sum(1 for kw in keywords if self._normalize_text(kw) in gt_normalized)
            gen_count = sum(1 for kw in keywords if self._normalize_text(kw) in gen_normalized)

            # Find which specific keywords are in ground truth
            gt_keywords = set([kw for kw in keywords if self._normalize_text(kw) in gt_normalized])
            gen_keywords = set([kw for kw in keywords if self._normalize_text(kw) in gen_normalized])

            if len(gt_keywords) > 0:
                # Calculate recall: how many GT keywords are also in generated text
                matched = len(gt_keywords & gen_keywords)
                recalls[f'{category}_recall'] = matched / len(gt_keywords)
                recalls[f'{category}_precision'] = matched / len(gen_keywords) if len(gen_keywords) > 0 else 0
            else:
                # No keywords from this category in ground truth
                recalls[f'{category}_recall'] = None
                recalls[f'{category}_precision'] = None

        # Overall entity recall (average of non-None recall values)
        valid_recalls = [v for k, v in recalls.items() if v is not None and 'recall' in k]
        recalls['overall_entity_recall'] = np.mean(valid_recalls) if valid_recalls else 0.0

        return recalls

    def extract_numerical_info(self, text: str) -> Dict[str, List]:
        """Extract numerical information with context"""
        numbers = {}

        patterns = {
            'blood_pressure': r'血圧[：:\s]*(\d+)\s*/\s*(\d+)',
            'pulse': r'脈拍[：:\s]*(\d+)',
            'temperature': r'体温[：:\s]*(\d+\.?\d*)',
            'spo2': r'[Ss][Pp][Oo]2?[：:\s]*(\d+)',
            'weight': r'体重[：:\s]*(\d+\.?\d*)\s*kg',
            'dosage': r'(\d+)\s*錠',
            'dosage_mg': r'(\d+)\s*mg',
            'date': r'(\d{1,2})[/月](\d{1,2})',
            'time': r'(\d{1,2})[時:](\d{1,2})?'
        }

        for key, pattern in patterns.items():
            matches = re.findall(pattern, text)
            numbers[key] = matches

        return numbers

    def calculate_numerical_accuracy(self, ground_truth: str, generated: str) -> Dict[str, float]:
        """Calculate accuracy of numerical information"""
        gt_numbers = self.extract_numerical_info(ground_truth)
        gen_numbers = self.extract_numerical_info(generated)

        accuracies = {}
        for category in gt_numbers.keys():
            gt_nums = set([self._normalize_number(n) for n in gt_numbers[category]])
            gen_nums = set([self._normalize_number(n) for n in gen_numbers[category]])

            if len(gt_nums) > 0:
                matched = len(gt_nums & gen_nums)
                accuracies[f'{category}_recall'] = matched / len(gt_nums)
                accuracies[f'{category}_precision'] = matched / len(gen_nums) if len(gen_nums) > 0 else 0

                # Check for hallucinated numbers
                hallucinated = gen_nums - gt_nums
                accuracies[f'{category}_hallucination_count'] = len(hallucinated)
            else:
                accuracies[f'{category}_recall'] = None
                accuracies[f'{category}_precision'] = None
                accuracies[f'{category}_hallucination_count'] = 0

        # Overall numerical accuracy
        valid_recalls = [v for k, v in accuracies.items() if v is not None and 'recall' in k]
        accuracies['overall_numerical_recall'] = np.mean(valid_recalls) if valid_recalls else 0.0

        return accuracies

    def check_section_completeness(self, text: str) -> Dict:
        """Check if all required sections are present and complete"""
        present_sections = {}
        section_lengths = {}

        for section_name, pattern in self.required_sections.items():
            match = re.search(pattern, text)
            present_sections[section_name] = match is not None

            if match:
                # Try to extract section content
                content_pattern = pattern + r'[：:\s]*(.*?)(?=【|<|$)'
                content_match = re.search(content_pattern, text, re.DOTALL)
                if content_match and content_match.group(1) is not None:
                    content = content_match.group(1).strip()
                    section_lengths[section_name] = len(content)
                else:
                    section_lengths[section_name] = 0
            else:
                section_lengths[section_name] = 0

        # Calculate scores
        presence_score = sum(present_sections.values()) / len(self.required_sections)

        # Completeness: section present AND has meaningful content (>20 chars)
        complete_sections = sum(1 for name, present in present_sections.items()
                               if present and section_lengths[name] > 20)
        completeness_score = complete_sections / len(self.required_sections)

        # List missing sections
        missing_sections = [name for name, present in present_sections.items() if not present]
        incomplete_sections = [name for name, present in present_sections.items()
                             if present and section_lengths[name] <= 20]

        return {
            'presence_score': presence_score,
            'completeness_score': completeness_score,
            'complete_section_count': complete_sections,
            'missing_sections': missing_sections,
            'incomplete_sections': incomplete_sections,
            'section_details': {name: {'present': present_sections[name],
                                      'length': section_lengths[name]}
                              for name in self.required_sections.keys()}
        }

    def calculate_coverage_density(self, source_text: str, summary_text: str) -> Dict[str, float]:
        """Calculate content coverage and density"""

        # Tokenize
        source_tokens = [token.surface for token in self.tokenizer.tokenize(source_text)
                        if len(token.surface.strip()) > 0]
        summary_tokens = [token.surface for token in self.tokenizer.tokenize(summary_text)
                         if len(token.surface.strip()) > 0]

        # Remove punctuation
        source_tokens = [t for t in source_tokens if not re.match(r'^[、。！？\s]+$', t)]
        summary_tokens = [t for t in summary_tokens if not re.match(r'^[、。！？\s]+$', t)]

        # Calculate metrics
        source_set = set(source_tokens)
        summary_set = set(summary_tokens)

        coverage = len(source_set & summary_set) / len(source_set) if len(source_set) > 0 else 0

        matched_source = [t for t in source_tokens if t in summary_set]
        density = len(matched_source) / len(summary_tokens) if len(summary_tokens) > 0 else 0

        compression_ratio = len(summary_tokens) / len(source_tokens) if len(source_tokens) > 0 else 0

        return {
            'coverage': coverage,
            'density': density,
            'compression_ratio': compression_ratio,
            'source_vocab_size': len(source_set),
            'summary_vocab_size': len(summary_set),
            'novel_terms': len(summary_set - source_set)
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        # Remove extra spaces and punctuation
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[、。！？]', '', text)
        return text.lower()

    @staticmethod
    def _normalize_number(num_tuple) -> str:
        """Normalize number tuple to string"""
        if isinstance(num_tuple, tuple):
            return '_'.join(str(n) for n in num_tuple)
        return str(num_tuple)

    def calculate_semantic_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Calculate semantic similarity between two texts using multiple approaches

        Args:
            text1: First text (typically ground truth)
            text2: Second text (typically generated summary)

        Returns:
            Dictionary with similarity scores
        """
        # 1. Character-based similarity (handles Japanese well)
        char_similarity = SequenceMatcher(None, text1, text2).ratio()

        # 2. Token-based Jaccard similarity
        tokens1 = set([token.surface for token in self.tokenizer.tokenize(text1)
                      if len(token.surface.strip()) > 0])
        tokens2 = set([token.surface for token in self.tokenizer.tokenize(text2)
                      if len(token.surface.strip()) > 0])

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        jaccard_similarity = intersection / union if union > 0 else 0.0

        # 3. TF-IDF cosine similarity
        try:
            # Custom tokenizer for TfidfVectorizer
            def japanese_tokenizer(text):
                return [token.surface for token in self.tokenizer.tokenize(text)
                       if len(token.surface.strip()) > 0]

            vectorizer = TfidfVectorizer(
                tokenizer=japanese_tokenizer,
                lowercase=False,
                min_df=1,
                token_pattern=None  # Suppress warning when using custom tokenizer
            )

            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception as e:
            # Fallback if TF-IDF fails (e.g., empty texts)
            cosine_sim = 0.0

        # 4. N-gram overlap (character-level for Japanese)
        def get_ngrams(text, n=3):
            """Extract character n-grams"""
            text = re.sub(r'\s+', '', text)  # Remove whitespace
            return set([text[i:i+n] for i in range(len(text)-n+1)])

        trigrams1 = get_ngrams(text1, 3)
        trigrams2 = get_ngrams(text2, 3)

        trigram_intersection = len(trigrams1 & trigrams2)
        trigram_union = len(trigrams1 | trigrams2)
        trigram_similarity = trigram_intersection / trigram_union if trigram_union > 0 else 0.0

        # 5. Combined semantic score (weighted average)
        weights = {
            'char': 0.2,
            'jaccard': 0.25,
            'cosine': 0.35,
            'trigram': 0.2
        }

        combined_score = (
            weights['char'] * char_similarity +
            weights['jaccard'] * jaccard_similarity +
            weights['cosine'] * cosine_sim +
            weights['trigram'] * trigram_similarity
        )

        return {
            'character_similarity': char_similarity,
            'jaccard_similarity': jaccard_similarity,
            'tfidf_cosine_similarity': cosine_sim,
            'trigram_similarity': trigram_similarity,
            'combined_semantic_score': combined_score,
            'token_overlap_count': intersection,
            'token_union_count': union
        }

    def calculate_rouge_scores(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores with DUAL tokenization (character + word level)

        Args:
            reference: Ground truth text
            hypothesis: Generated summary

        Returns:
            Dictionary with ROUGE scores for both character and word tokenization
        """
        if not ROUGE_AVAILABLE:
            return {
                'rouge1_f1': None, 'rouge1_precision': None, 'rouge1_recall': None,
                'rouge2_f1': None, 'rouge2_precision': None, 'rouge2_recall': None,
                'rougeL_f1': None, 'rougeL_precision': None, 'rougeL_recall': None,
                'rouge1_f1_char': None, 'rouge2_f1_char': None, 'rougeL_f1_char': None,
                'rouge1_f1_word': None, 'rouge2_f1_word': None, 'rougeL_f1_word': None,
                'error': 'rouge-score not installed'
            }

        try:
            # Word-level ROUGE (default, using built-in tokenizer)
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
            word_scores = scorer.score(reference, hypothesis)

            # Character-level ROUGE using custom tokenizer class
            class CharTokenizer:
                """Character-level tokenizer for Japanese"""
                def tokenize(self, text):
                    return list(text.replace(' ', '').replace('\n', ''))

            char_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=False,
                tokenizer=CharTokenizer()
            )
            char_scores = char_scorer.score(reference, hypothesis)

            return {
                # Word-level scores (default)
                'rouge1_f1': word_scores['rouge1'].fmeasure,
                'rouge1_precision': word_scores['rouge1'].precision,
                'rouge1_recall': word_scores['rouge1'].recall,
                'rouge2_f1': word_scores['rouge2'].fmeasure,
                'rouge2_precision': word_scores['rouge2'].precision,
                'rouge2_recall': word_scores['rouge2'].recall,
                'rougeL_f1': word_scores['rougeL'].fmeasure,
                'rougeL_precision': word_scores['rougeL'].precision,
                'rougeL_recall': word_scores['rougeL'].recall,

                # Character-level scores
                'rouge1_f1_char': char_scores['rouge1'].fmeasure,
                'rouge2_f1_char': char_scores['rouge2'].fmeasure,
                'rougeL_f1_char': char_scores['rougeL'].fmeasure,

                # Word-level scores (explicit)
                'rouge1_f1_word': word_scores['rouge1'].fmeasure,
                'rouge2_f1_word': word_scores['rouge2'].fmeasure,
                'rougeL_f1_word': word_scores['rougeL'].fmeasure
            }
        except Exception as e:
            return {
                'rouge1_f1': None, 'rouge1_precision': None, 'rouge1_recall': None,
                'rouge2_f1': None, 'rouge2_precision': None, 'rouge2_recall': None,
                'rougeL_f1': None, 'rougeL_precision': None, 'rougeL_recall': None,
                'error': str(e)
            }

    def calculate_bleu_score(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate BLEU score with DUAL tokenization (character + word level)

        Args:
            reference: Ground truth text
            hypothesis: Generated summary

        Returns:
            Dictionary with BLEU scores for both character and word tokenization
        """
        # Dynamic check in case sacrebleu was installed after initial import
        if not check_bleu_available():
            return {
                'bleu': None, 'bleu_char': None, 'bleu_word': None,
                'bleu_1': None, 'bleu_2': None, 'bleu_3': None, 'bleu_4': None,
                'error': 'sacrebleu not installed'
            }

        try:
            import sacrebleu

            # ===== WORD-LEVEL BLEU (using Janome tokenizer) =====
            ref_words = [token.surface for token in self.tokenizer.tokenize(reference)]
            hyp_words = [token.surface for token in self.tokenizer.tokenize(hypothesis)]

            # Join tokens with spaces for sacrebleu
            ref_word_str = ' '.join(ref_words) if ref_words else ''
            hyp_word_str = ' '.join(hyp_words) if hyp_words else ''

            bleu_word = sacrebleu.corpus_bleu([hyp_word_str], [[ref_word_str]], tokenize='none')

            # ===== CHARACTER-LEVEL BLEU =====
            # Remove spaces and newlines for character-level comparison
            ref_chars = list(reference.replace(' ', '').replace('\n', ''))
            hyp_chars = list(hypothesis.replace(' ', '').replace('\n', ''))

            ref_char_str = ' '.join(ref_chars) if ref_chars else ''
            hyp_char_str = ' '.join(hyp_chars) if hyp_chars else ''

            bleu_char = sacrebleu.corpus_bleu([hyp_char_str], [[ref_char_str]], tokenize='none')

            # ===== DEFAULT BLEU (word-level, for backward compatibility) =====
            bleu = bleu_word

            # Individual n-gram precisions (word-level)
            bleu_1 = sacrebleu.BLEU(max_ngram_order=1).corpus_score([hyp_word_str], [[ref_word_str]])
            bleu_2 = sacrebleu.BLEU(max_ngram_order=2).corpus_score([hyp_word_str], [[ref_word_str]])
            bleu_3 = sacrebleu.BLEU(max_ngram_order=3).corpus_score([hyp_word_str], [[ref_word_str]])
            bleu_4 = sacrebleu.BLEU(max_ngram_order=4).corpus_score([hyp_word_str], [[ref_word_str]])

            return {
                # Default (word-level for backward compatibility)
                'bleu': bleu.score / 100.0,
                'bleu_1': bleu_1.score / 100.0,
                'bleu_2': bleu_2.score / 100.0,
                'bleu_3': bleu_3.score / 100.0,
                'bleu_4': bleu_4.score / 100.0,
                'bp': bleu.bp,
                'sys_len': bleu.sys_len,
                'ref_len': bleu.ref_len,

                # Character-level BLEU
                'bleu_char': bleu_char.score / 100.0,
                'bleu_char_bp': bleu_char.bp,

                # Word-level BLEU (explicit)
                'bleu_word': bleu_word.score / 100.0,
                'bleu_word_bp': bleu_word.bp
            }
        except Exception as e:
            return {
                'bleu': None, 'bleu_1': None, 'bleu_2': None,
                'bleu_3': None, 'bleu_4': None, 'error': str(e)
            }

    def calculate_bertscore(self, reference: str, hypothesis: str,
                           lang: str = 'ja', model_type: str = None) -> Dict[str, float]:
        """
        Calculate BERTScore using contextual embeddings

        Args:
            reference: Ground truth text
            hypothesis: Generated summary
            lang: Language code (default: 'ja' for Japanese)
            model_type: Specific model to use (e.g., 'cl-tohoku/bert-base-japanese-v3')

        Returns:
            Dictionary with BERTScore precision, recall, F1
        """
        # Dynamic check in case bert-score was installed after initial import
        if not check_bertscore_available():
            return {
                'bertscore_precision': None,
                'bertscore_recall': None,
                'bertscore_f1': None,
                'error': 'bert-score not installed'
            }

        try:
            from bert_score import score as bert_score

            # Use Japanese-specific model if not specified
            if model_type is None:
                # Default to Japanese BERT (better than multilingual for Japanese)
                model_type = 'xlm-roberta-large'  # Alternatively, use 'cl-tohoku/bert-base-japanese-v3'

            # Calculate BERTScore
            # Note: When using custom model_type, we don't pass lang parameter
            # The model itself determines the language handling
            P, R, F1 = bert_score(
                [hypothesis],
                [reference],
                model_type=model_type,
                verbose=False,
                num_layers=9  # Use middle layers (works well for most BERT models)
            )

            return {
                'bertscore_precision': float(P.mean()),
                'bertscore_recall': float(R.mean()),
                'bertscore_f1': float(F1.mean()),
                'model_type': model_type
            }
        except Exception as e:
            return {
                'bertscore_precision': None,
                'bertscore_recall': None,
                'bertscore_f1': None,
                'error': str(e)
            }

    def calculate_cosine_similarity_simple(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Calculate TF-IDF based cosine similarity (standalone method)

        This is extracted from calculate_semantic_similarity for direct use.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Dictionary with cosine similarity score
        """
        try:
            # Custom tokenizer for TfidfVectorizer
            def japanese_tokenizer(text):
                return [token.surface for token in self.tokenizer.tokenize(text)
                       if len(token.surface.strip()) > 0]

            vectorizer = TfidfVectorizer(
                tokenizer=japanese_tokenizer,
                lowercase=False,
                min_df=1,
                token_pattern=None  # Suppress warning when using custom tokenizer
            )

            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            # Also calculate vocabulary overlap
            vocab = vectorizer.get_feature_names_out()
            tfidf_array = tfidf_matrix.toarray()

            # Non-zero features in each document
            text1_features = set(vocab[tfidf_array[0] > 0])
            text2_features = set(vocab[tfidf_array[1] > 0])

            vocab_overlap = len(text1_features & text2_features)
            vocab_union = len(text1_features | text2_features)

            return {
                'cosine_similarity': float(cosine_sim),
                'vocab_overlap': vocab_overlap,
                'vocab_union': vocab_union,
                'vocab_overlap_ratio': vocab_overlap / vocab_union if vocab_union > 0 else 0.0,
                'text1_vocab_size': len(text1_features),
                'text2_vocab_size': len(text2_features)
            }
        except Exception as e:
            return {
                'cosine_similarity': 0.0,
                'vocab_overlap': 0,
                'vocab_union': 0,
                'vocab_overlap_ratio': 0.0,
                'error': str(e)
            }


class StatisticalTests:
    """Statistical significance testing"""

    @staticmethod
    def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
        """
        Apply Bonferroni correction for multiple comparisons

        Args:
            p_values: List of p-values from multiple tests
            alpha: Significance level (default 0.05)

        Returns:
            Dictionary with corrected p-values and significance decisions
        """
        n_tests = len(p_values)
        corrected_alpha = alpha / n_tests

        corrected_results = []
        for i, p_val in enumerate(p_values):
            is_significant = p_val < corrected_alpha

            # Determine significance level with Bonferroni correction
            if p_val < corrected_alpha:
                if p_val < 0.001 / n_tests:
                    sig_label = "***"
                elif p_val < 0.01 / n_tests:
                    sig_label = "**"
                elif p_val < 0.05 / n_tests:
                    sig_label = "*"
                else:
                    sig_label = "*"
            else:
                sig_label = "ns"

            corrected_results.append({
                'original_p_value': p_val,
                'corrected_alpha': corrected_alpha,
                'is_significant': is_significant,
                'significance_label': sig_label
            })

        return {
            'n_comparisons': n_tests,
            'original_alpha': alpha,
            'corrected_alpha': corrected_alpha,
            'results': corrected_results,
            'n_significant': sum(r['is_significant'] for r in corrected_results)
        }

    @staticmethod
    def paired_t_test(system_a: List[float], system_b: List[float],
                     metric_name: str = "metric") -> Dict:
        """Perform paired t-test between two systems"""

        # Paired t-test
        t_stat, p_value_t = stats.ttest_rel(system_a, system_b)

        # Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, p_value_w = stats.wilcoxon(system_a, system_b)
        except:
            p_value_w = None
            w_stat = None

        # Effect size (Cohen's d)
        mean_diff = np.mean(system_a) - np.mean(system_b)
        pooled_std = np.sqrt((np.std(system_a, ddof=1)**2 + np.std(system_b, ddof=1)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        # Confidence interval
        diff = np.array(system_a) - np.array(system_b)
        ci = stats.t.interval(0.95, len(diff)-1,
                            loc=np.mean(diff),
                            scale=stats.sem(diff))

        # Interpret effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect_size_interpretation = "negligible"
        elif abs_d < 0.5:
            effect_size_interpretation = "small"
        elif abs_d < 0.8:
            effect_size_interpretation = "medium"
        else:
            effect_size_interpretation = "large"

        # Interpret p-value
        if p_value_t < 0.001:
            significance = "***"
        elif p_value_t < 0.01:
            significance = "**"
        elif p_value_t < 0.05:
            significance = "*"
        else:
            significance = "ns"

        return {
            'metric': metric_name,
            'mean_system_a': np.mean(system_a),
            'mean_system_b': np.mean(system_b),
            'std_system_a': np.std(system_a, ddof=1),
            'std_system_b': np.std(system_b, ddof=1),
            'mean_difference': mean_diff,
            't_statistic': t_stat,
            'p_value_t': p_value_t,
            'p_value_wilcoxon': p_value_w,
            'cohens_d': cohens_d,
            'effect_size': effect_size_interpretation,
            'ci_95_lower': ci[0],
            'ci_95_upper': ci[1],
            'significance': significance,
            'n_samples': len(system_a)
        }

    @staticmethod
    def format_results_table(results_list: List[Dict]) -> pd.DataFrame:
        """Format statistical test results as table"""
        df = pd.DataFrame(results_list)

        # Reorder columns
        column_order = ['metric', 'mean_system_a', 'mean_system_b', 'mean_difference',
                       'p_value_t', 'significance', 'cohens_d', 'effect_size',
                       'ci_95_lower', 'ci_95_upper']

        df = df[column_order]

        # Round numeric columns
        numeric_cols = ['mean_system_a', 'mean_system_b', 'mean_difference',
                       'p_value_t', 'cohens_d', 'ci_95_lower', 'ci_95_upper']
        df[numeric_cols] = df[numeric_cols].round(4)

        return df


# Example usage
if __name__ == "__main__":
    print("Clinical Metrics Module")
    print("=" * 60)

    # Example texts
    ground_truth = """
    【利用者の状態（病状・トピックス）】血圧は112/64、脈拍は63、体温は36.5度。
    排便は毎日みられている。コーラック2錠服用。右下肢の痺れあり。
    <処置内容・訓練内容>バイタルチェック、状態観察。
    <備考>特記事項なし。
    <外部連絡・伝達事項>主治医に報告済み。
    """

    rag_summary = """
    利用者の状態（状態・トピックス）
    【トピックス】血圧112/64、脈拍63、体温36.5度で安定。右下肢の痺れ継続。
    排便は毎日あり、コーラック2錠服用中。
    <処置内容・訓練内容>バイタルチェックと状態観察を実施。
    <備考>特記事項なし。
    <外部連絡・伝達事項>主治医に状態報告済み。
    """

    multi_summary = """
    利用者の状態は安定。血圧112/64。排便あり、内服継続。
    処置内容: 状態観察。
    """

    # Initialize metrics
    metrics = ClinicalMetrics()

    print("\n1. Entity Recall")
    print("-" * 60)
    entity_recall_rag = metrics.calculate_entity_recall(ground_truth, rag_summary)
    entity_recall_multi = metrics.calculate_entity_recall(ground_truth, multi_summary)
    print(f"RAG Overall Entity Recall: {entity_recall_rag['overall_entity_recall']:.3f}")
    print(f"Multi-stage Overall Entity Recall: {entity_recall_multi['overall_entity_recall']:.3f}")

    print("\n2. Numerical Accuracy")
    print("-" * 60)
    num_acc_rag = metrics.calculate_numerical_accuracy(ground_truth, rag_summary)
    num_acc_multi = metrics.calculate_numerical_accuracy(ground_truth, multi_summary)
    print(f"RAG Numerical Recall: {num_acc_rag['overall_numerical_recall']:.3f}")
    print(f"Multi-stage Numerical Recall: {num_acc_multi['overall_numerical_recall']:.3f}")

    print("\n3. Section Completeness")
    print("-" * 60)
    section_rag = metrics.check_section_completeness(rag_summary)
    section_multi = metrics.check_section_completeness(multi_summary)
    print(f"RAG Completeness: {section_rag['completeness_score']:.3f}")
    print(f"Multi-stage Completeness: {section_multi['completeness_score']:.3f}")
    print(f"RAG Missing sections: {section_rag['missing_sections']}")
    print(f"Multi-stage Missing sections: {section_multi['missing_sections']}")

    print("\n4. Semantic Similarity")
    print("-" * 60)
    semantic_rag = metrics.calculate_semantic_similarity(ground_truth, rag_summary)
    semantic_multi = metrics.calculate_semantic_similarity(ground_truth, multi_summary)
    print(f"RAG Combined Semantic Score: {semantic_rag['combined_semantic_score']:.3f}")
    print(f"  - TF-IDF Cosine: {semantic_rag['tfidf_cosine_similarity']:.3f}")
    print(f"  - Jaccard: {semantic_rag['jaccard_similarity']:.3f}")
    print(f"  - Character: {semantic_rag['character_similarity']:.3f}")
    print(f"Multi-stage Combined Semantic Score: {semantic_multi['combined_semantic_score']:.3f}")
    print(f"  - TF-IDF Cosine: {semantic_multi['tfidf_cosine_similarity']:.3f}")
    print(f"  - Jaccard: {semantic_multi['jaccard_similarity']:.3f}")
    print(f"  - Character: {semantic_multi['character_similarity']:.3f}")

    print("\n5. Statistical Tests with Bonferroni Correction")
    print("-" * 60)
    # Simulate some test results
    test_results = [
        StatisticalTests.paired_t_test([0.8, 0.9, 0.85], [0.7, 0.75, 0.72], "Metric 1"),
        StatisticalTests.paired_t_test([0.6, 0.65, 0.7], [0.55, 0.6, 0.65], "Metric 2"),
        StatisticalTests.paired_t_test([0.5, 0.52, 0.48], [0.49, 0.51, 0.47], "Metric 3"),
    ]

    p_values = [t['p_value_t'] for t in test_results]
    bonferroni = StatisticalTests.bonferroni_correction(p_values)

    print(f"Number of tests: {bonferroni['n_comparisons']}")
    print(f"Corrected alpha: {bonferroni['corrected_alpha']:.6f}")
    print(f"Significant after correction: {bonferroni['n_significant']}")

    for test, corr in zip(test_results, bonferroni['results']):
        print(f"\n{test['metric']}:")
        print(f"  Original p-value: {corr['original_p_value']:.6f}")
        print(f"  Significant: {corr['is_significant']} ({corr['significance_label']})")

    print("\n" + "=" * 60)
    print("✓ Module loaded successfully")
