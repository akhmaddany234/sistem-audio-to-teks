# utils/summarizer.py (versi sederhana - TANPA transformers)
import nltk
from nltk.tokenize import sent_tokenize
import re
from collections import Counter
from .corrections import correct_keywords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Daftar stopwords sederhana untuk bahasa Indonesia
STOPWORDS = set([
    'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'pada', 'adalah', 'dengan',
    'ini', 'itu', 'juga', 'akan', 'telah', 'sudah', 'bisa', 'dapat', 'harus',
    'atau', 'karena', 'oleh', 'saat', 'sebagai', 'mereka', 'kami', 'kita',
    'saya', 'anda', 'dia', 'itu', 'ada', 'ya', 'tidak', 'yaitu', 'yakni',
    'tersebut', 'merupakan', 'sehingga', 'namun', 'tetapi', 'walaupun',
    'meskipun', 'kalau', 'jika', 'apabila', 'maka', 'mari', 'ayo', 'silahkan'
])

def load_nlp():
    """Fungsi sederhana untuk kompatibilitas dengan kode lama"""
    return None

def extract_keywords_simple(text, num_keywords=20):
    """Ekstrak kata kunci sederhana tanpa spaCy"""
    words = text.lower().split()
    
    cleaned_words = []
    for word in words:
        word = word.strip('.,;:!?()[]{}"\'')
        if len(word) > 3 and word not in STOPWORDS:
            cleaned_words.append(word)
    
    word_freq = Counter(cleaned_words)
    top_keywords = [word for word, _ in word_freq.most_common(num_keywords)]
    
    return top_keywords

def generate_summary(text):
    """Versi sederhana generate summary"""
    try:
        sentences = sent_tokenize(text)
        summary = ' '.join(sentences[:5]) if sentences else text[:500]
        
        # Cari poin keputusan
        key_points = []
        decision_words = ['memutuskan', 'keputusan', 'setuju', 'disetujui', 'menyetujui', 'sepakat']
        
        for sent in sentences:
            if any(word in sent.lower() for word in decision_words):
                key_points.append(sent)
        
        # Cari timeline (tanggal, deadline)
        timeline = []
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{1,2}\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)',
            r'(?:senin|selasa|rabu|kamis|jumat|sabtu|minggu)',
            r'(?:besok|minggu depan|bulan depan)'
        ]
        
        for sent in sentences:
            if 'deadline' in sent.lower() or 'tenggat' in sent.lower() or 'target' in sent.lower():
                for pattern in date_patterns:
                    if re.search(pattern, sent.lower()):
                        timeline.append(sent)
                        break
        
        return {
            'summary': summary,
            'key_points': key_points[:5],
            'timeline': timeline[:3],
            'full_text': text[:500] + "..." if len(text) > 500 else text
        }
        
    except Exception as e:
        return {
            'summary': text[:500] + "..." if len(text) > 500 else text,
            'key_points': [],
            'timeline': [],
            'full_text': text[:500] + "..." if len(text) > 500 else text
        }

def extract_keywords(text, num_keywords=20):
    """Ekstrak kata kunci penting dari teks"""
    try:
        top_keywords = extract_keywords_simple(text, num_keywords)
        
        try:
            corrected_keywords = correct_keywords(top_keywords)
            return corrected_keywords
        except:
            return top_keywords
    
    except Exception as e:
        words = text.lower().split()
        words = [w.strip('.,;:!?()[]{}"\'') for w in words if len(w) > 3]
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(num_keywords)]