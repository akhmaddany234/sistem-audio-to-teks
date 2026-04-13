# utils/summarizer_advanced.py
"""
Advanced Summarization menggunakan Transformer Models (IndoBERT)
Untuk menghasilkan ringkasan yang lebih akurat dan natural
"""

import streamlit as st
from nltk.tokenize import sent_tokenize
import re
from collections import Counter
import numpy as np

try:
    from transformers import (
        pipeline,                    # ✅ Pipeline untuk summarization
        AutoTokenizer,               # ✅ Tokenizer otomatis
        AutoModelForSeq2SeqLM,       # ✅ Model sequence-to-sequence
        AutoModelForSequenceClassification  # ✅ Untuk zero-shot
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    st.error(f"❌ Transformers library tidak tersedia: {e}")
    st.info("Install dengan: pip install transformers torch")

# Import dari file yang sudah ada untuk fallback
from .corrections import correct_keywords
from .summarizer import STOPWORDS, extract_keywords_simple


# Daftar stopwords (sama dengan summarizer.py)
STOPWORDS = STOPWORDS

@st.cache_resource
def load_summarizer_model(model_name="cahya/bert2bert-indonesian-summarization"):
    """
    Load IndoBERT summarization model (cached)
    
    Model options:
    - "cahya/bert2bert-indonesian-summarization" (recommended for Indonesian)
    - "indonesian-nlp/indobert-summarization" (alternative)
    - "facebook/bart-large-cnn" (English, fallback)
    """
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        with st.spinner(f"🔄 Memuat model summarization {model_name}..."):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                framework="pt"
            )
            return summarizer
    except Exception as e:
        st.warning(f"Gagal load model {model_name}: {e}")
        return None

@st.cache_resource
def load_zero_shot_classifier():
    """Load zero-shot classification model untuk key points extraction"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        return classifier
    except Exception as e:
        st.warning(f"Gagal load zero-shot classifier: {e}")
        return None

@st.cache_resource
def load_ner_model():
    """Load NER model untuk timeline extraction"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        # Gunakan IndoBERT untuk NER
        ner = pipeline(
            "ner",
            model="indobenchmark/indobert-base-p1",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        return ner
    except Exception as e:
        st.warning(f"Gagal load NER model: {e}")
        return None

class AdvancedSummarizer:
    """
    Advanced summarizer dengan transformer models
    """
    
    def __init__(self):
        self.summarizer = load_summarizer_model()
        self.classifier = load_zero_shot_classifier()
        self.ner = load_ner_model()
        
        # Fallback ke simple summarizer jika model tidak tersedia
        self.use_fallback = self.summarizer is None
    
    def summarize_abstractive(self, text: str, max_length: int = 200, min_length: int = 50) -> str:
        """
        Abstractive summarization dengan transformer
        
        Parameters:
        - text: teks yang akan diringkas
        - max_length: panjang maksimum ringkasan (tokens)
        - min_length: panjang minimum ringkasan (tokens)
        """
        if self.use_fallback or not text:
            return self._extractive_fallback(text)
        
        try:
            # Batasi input untuk model (token limit ~1024)
            input_text = text[:3000]  # Perkiraan token limit
            
            summary = self.summarizer(
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=4,
                early_stopping=True
            )
            
            return summary[0]['summary_text']
            
        except Exception as e:
            st.warning(f"Abstractive summarization failed: {e}")
            return self._extractive_fallback(text)
    
    def _extractive_fallback(self, text: str) -> str:
        """Fallback ke extractive summarization (5 kalimat pertama)"""
        try:
            sentences = sent_tokenize(text)
            return ' '.join(sentences[:5]) if sentences else text[:500]
        except:
            return text[:500] + "..." if len(text) > 500 else text
    
    def extract_key_points_advanced(self, text: str) -> list:
        """
        Ekstrak key points dengan zero-shot classification
        Mengidentifikasi kalimat yang mengandung keputusan penting
        """
        if not self.classifier:
            return self._extract_key_points_fallback(text)
        
        try:
            # Kategori yang ingin dideteksi
            candidate_labels = [
                "keputusan rapat", "tindak lanjut", "deadline", 
                "masalah yang dibahas", "solusi yang disepakati", 
                "rekomendasi", "kesepakatan"
            ]
            
            # Tokenisasi kalimat
            sentences = sent_tokenize(text)
            key_points = []
            
            # Proses maksimal 20 kalimat untuk performa
            for sent in sentences[:20]:
                # Skip kalimat pendek
                if len(sent.split()) < 5:
                    continue
                
                # Klasifikasi
                result = self.classifier(sent, candidate_labels)
                
                # Jika confidence tinggi dan termasuk kategori keputusan
                top_label = result['labels'][0]
                top_score = result['scores'][0]
                
                if top_score > 0.7 and top_label in ['keputusan rapat', 'kesepakatan', 'tindak lanjut']:
                    key_points.append(sent.strip())
            
            return key_points[:5]  # Max 5 poin
            
        except Exception as e:
            st.warning(f"Key points extraction failed: {e}")
            return self._extract_key_points_fallback(text)
    
    def _extract_key_points_fallback(self, text: str) -> list:
        """Fallback untuk key points (keyword-based)"""
        decision_words = ['memutuskan', 'keputusan', 'setuju', 'disetujui', 'menyetujui', 'sepakat']
        sentences = sent_tokenize(text)
        key_points = []
        
        for sent in sentences:
            if any(word in sent.lower() for word in decision_words):
                key_points.append(sent)
        
        return key_points[:5]
    
    def extract_timeline_advanced(self, text: str) -> list:
        """
        Extract timeline dengan NER (Named Entity Recognition)
        Mendeteksi tanggal dan deadline
        """
        if not self.ner:
            return self._extract_timeline_regex(text)
        
        try:
            # Tokenisasi kalimat
            sentences = sent_tokenize(text)
            timeline = []
            
            for sent in sentences:
                # Cek apakah mengandung kata deadline/tenggat
                has_deadline = any(kw in sent.lower() for kw in ['deadline', 'tenggat', 'target', 'selesai', 'dikerjakan'])
                
                if has_deadline:
                    # Ekstrak entities dengan NER
                    entities = self.ner(sent)
                    
                    # Cek apakah ada entity DATE
                    has_date = any(e.get('entity_group') == 'DATE' or e.get('entity') in ['B-DATE', 'I-DATE'] for e in entities)
                    
                    if has_date:
                        timeline.append(sent.strip())
            
            return timeline[:3]  # Max 3 timeline
            
        except Exception as e:
            st.warning(f"Timeline extraction failed: {e}")
            return self._extract_timeline_regex(text)
    
    def _extract_timeline_regex(self, text: str) -> list:
        """Fallback timeline extraction dengan regex"""
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # 31/12/2024
            r'\d{1,2}\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)',  # 31 Desember 2024
            r'(?:senin|selasa|rabu|kamis|jumat|sabtu|minggu)',  # hari
            r'(?:besok|minggu depan|bulan depan|tahun depan)'  # relatif
        ]
        
        sentences = sent_tokenize(text)
        timeline = []
        
        for sent in sentences:
            if any(kw in sent.lower() for kw in ['deadline', 'tenggat', 'target', 'selesai']):
                for pattern in date_patterns:
                    if re.search(pattern, sent.lower()):
                        timeline.append(sent.strip())
                        break
        
        return timeline[:3]
    
    def extract_keywords_advanced(self, text: str, num_keywords: int = 20) -> list:
        """
        Ekstrak keywords dengan TF-IDF atau simple frequency
        Bisa ditingkatkan dengan keyBERT nanti
        """
        # Gunakan metode sederhana dari summarizer.py
        keywords = extract_keywords_simple(text, num_keywords)
        
        # Koreksi dengan kamus
        try:
            corrected = correct_keywords(keywords)
            return corrected
        except:
            return keywords
    
    def generate_complete_summary(self, text: str) -> dict:
        """
        Generate complete summary dengan semua komponen
        
        Returns:
        dict with keys: summary, key_points, timeline, full_text
        """
        if not text:
            return {
                'summary': 'Tidak ada teks untuk diringkas.',
                'key_points': [],
                'timeline': [],
                'full_text': ''
            }
        
        with st.spinner("🤖 Meringkas dengan AI (IndoBERT)..."):
            summary = self.summarize_abstractive(text)
        
        with st.spinner("📌 Mengekstrak poin keputusan..."):
            key_points = self.extract_key_points_advanced(text)
        
        with st.spinner("📅 Mengekstrak timeline..."):
            timeline = self.extract_timeline_advanced(text)
        
        with st.spinner("🔑 Mengekstrak kata kunci..."):
            keywords = self.extract_keywords_advanced(text)
        
        return {
            'summary': summary,
            'key_points': key_points,
            'timeline': timeline,
            'keywords': keywords,
            'full_text': text[:500] + "..." if len(text) > 500 else text
        }

# Fungsi wrapper untuk compatibility dengan kode lama
def generate_summary_advanced(text):
    """Wrapper untuk backward compatibility"""
    summarizer = AdvancedSummarizer()
    return summarizer.generate_complete_summary(text)

def extract_keywords_advanced(text, num_keywords=20):
    """Wrapper untuk ekstraksi keywords"""
    summarizer = AdvancedSummarizer()
    return summarizer.extract_keywords_advanced(text, num_keywords)