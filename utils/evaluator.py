# utils/evaluator.py
"""
Modul evaluasi untuk mengukur performa transkripsi dan summarization
Menggunakan metrik standar: WER, CER, ROUGE
"""

import streamlit as st
import numpy as np
import re

# Import untuk metrik
try:
    from jiwer import wer, cer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    st.warning("⚠️ Jiwer tidak terinstall. Install dengan: pip install jiwer")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    st.warning("⚠️ ROUGE Score tidak terinstall. Install dengan: pip install rouge-score")

@st.cache_data
def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER)
    Lower is better (0 = perfect)
    
    WER = (S + D + I) / N
    - S: Substitutions (kata salah)
    - D: Deletions (kata hilang)
    - I: Insertions (kata tambahan)
    - N: Total words in reference
    """
    if not JIWER_AVAILABLE:
        return _calculate_wer_manual(reference, hypothesis)
    
    try:
        # Bersihkan teks untuk perhitungan yang lebih akurat
        ref_clean = _clean_text(reference)
        hyp_clean = _clean_text(hypothesis)
        return wer(ref_clean, hyp_clean)
    except Exception as e:
        st.warning(f"Error calculating WER: {e}")
        return 1.0

@st.cache_data
def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER)
    Lower is better (0 = perfect)
    
    CER = (S + D + I) / N (dalam level karakter)
    """
    if not JIWER_AVAILABLE:
        return _calculate_cer_manual(reference, hypothesis)
    
    try:
        ref_clean = _clean_text(reference)
        hyp_clean = _clean_text(hypothesis)
        return cer(ref_clean, hyp_clean)
    except Exception as e:
        st.warning(f"Error calculating CER: {e}")
        return 1.0

@st.cache_data
def calculate_rouge_scores(reference: str, hypothesis: str) -> dict:
    """
    Calculate ROUGE scores for summarization
    Higher is better (max 1.0)
    
    Returns:
    - ROUGE-1: Unigram overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-L: Longest common subsequence
    """
    if not ROUGE_AVAILABLE:
        return _calculate_rouge_manual(reference, hypothesis)
    
    try:
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        scores = scorer.score(reference, hypothesis)
        
        return {
            'ROUGE-1': scores['rouge1'].fmeasure,
            'ROUGE-2': scores['rouge2'].fmeasure,
            'ROUGE-L': scores['rougeL'].fmeasure
        }
    except Exception as e:
        st.warning(f"Error calculating ROUGE: {e}")
        return _calculate_rouge_manual(reference, hypothesis)

def _clean_text(text: str) -> str:
    """Bersihkan teks untuk perhitungan metrik yang lebih akurat"""
    # Lowercase
    text = text.lower()
    # Hapus tanda baca
    text = re.sub(r'[^\w\s]', ' ', text)
    # Hapus multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _calculate_wer_manual(reference: str, hypothesis: str) -> float:
    """Manual WER calculation (fallback)"""
    ref_words = _clean_text(reference).split()
    hyp_words = _clean_text(hypothesis).split()
    
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    
    # Simple Levenshtein distance untuk kata
    n = len(ref_words)
    m = len(hyp_words)
    
    # Buat matrix DP
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    
    distance = dp[n][m]
    return distance / n

def _calculate_cer_manual(reference: str, hypothesis: str) -> float:
    """Manual CER calculation (fallback)"""
    ref_chars = list(_clean_text(reference).replace(' ', ''))
    hyp_chars = list(_clean_text(hypothesis).replace(' ', ''))
    
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    
    # Simple Levenshtein distance untuk karakter
    n = len(ref_chars)
    m = len(hyp_chars)
    
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    
    distance = dp[n][m]
    return distance / n

def _calculate_rouge_manual(reference: str, hypothesis: str) -> dict:
    """Manual ROUGE calculation sederhana (fallback)"""
    ref_words = set(_clean_text(reference).split())
    hyp_words = set(_clean_text(hypothesis).split())
    
    if not ref_words:
        return {'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0}
    
    # ROUGE-1: unigram overlap
    intersection = ref_words.intersection(hyp_words)
    rouge_1 = len(intersection) / len(ref_words)
    
    # ROUGE-2: bigram overlap (sederhana)
    ref_bigrams = set([f"{ref_words[i]} {ref_words[i+1]}" 
                       for i in range(len(ref_words)-1)]) if len(ref_words) > 1 else set()
    hyp_bigrams = set([f"{hyp_words[i]} {hyp_words[i+1]}" 
                       for i in range(len(hyp_words)-1)]) if len(hyp_words) > 1 else set()
    
    if ref_bigrams:
        bigram_intersection = ref_bigrams.intersection(hyp_bigrams)
        rouge_2 = len(bigram_intersection) / len(ref_bigrams)
    else:
        rouge_2 = 0.0
    
    return {
        'ROUGE-1': rouge_1,
        'ROUGE-2': rouge_2,
        'ROUGE-L': rouge_1  # Simplified
    }

def evaluate_transcription(ground_truth: str, transcript: str) -> dict:
    """
    Evaluasi lengkap untuk transkripsi
    """
    return {
        'wer': calculate_wer(ground_truth, transcript),
        'cer': calculate_cer(ground_truth, transcript),
        'reference_length': len(ground_truth.split()),
        'hypothesis_length': len(transcript.split())
    }

def evaluate_summary(ground_truth: str, summary: str) -> dict:
    """
    Evaluasi lengkap untuk ringkasan
    """
    return calculate_rouge_scores(ground_truth, summary)

def get_evaluation_interpretation(wer_score: float, rouge_scores: dict = None) -> dict:
    """
    Berikan interpretasi hasil evaluasi
    """
    # Interpretasi WER
    if wer_score < 0.1:
        wer_quality = "Sangat Baik ✅"
        wer_desc = "Transkripsi hampir sempurna, sangat akurat"
    elif wer_score < 0.2:
        wer_quality = "Baik 👍"
        wer_desc = "Transkripsi cukup akurat, sedikit error"
    elif wer_score < 0.3:
        wer_quality = "Cukup 📝"
        wer_desc = "Transkripsi lumayan, masih ada error signifikan"
    elif wer_score < 0.4:
        wer_quality = "Perlu Perbaikan ⚠️"
        wer_desc = "Banyak error, perlu peningkatan"
    else:
        wer_quality = "Kurang ❌"
        wer_desc = "Transkripsi tidak akurat, perlu perbaikan besar"
    
    result = {
        'wer_quality': wer_quality,
        'wer_description': wer_desc
    }
    
    # Interpretasi ROUGE jika ada
    if rouge_scores:
        rouge_1 = rouge_scores.get('ROUGE-1', 0)
        if rouge_1 > 0.7:
            result['rouge_quality'] = "Sangat Baik ✅"
            result['rouge_desc'] = "Ringkasan sangat baik, mempertahankan informasi penting"
        elif rouge_1 > 0.5:
            result['rouge_quality'] = "Baik 👍"
            result['rouge_desc'] = "Ringkasan baik, sebagian besar informasi penting tersampaikan"
        elif rouge_1 > 0.3:
            result['rouge_quality'] = "Cukup 📝"
            result['rouge_desc'] = "Ringkasan cukup, beberapa informasi penting hilang"
        else:
            result['rouge_quality'] = "Perlu Perbaikan ⚠️"
            result['rouge_desc'] = "Ringkasan kurang baik, banyak informasi penting hilang"
    
    return result