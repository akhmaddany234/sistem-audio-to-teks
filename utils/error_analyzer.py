# utils/error_analyzer.py
import pandas as pd
import numpy as np
from collections import Counter
import re

class ErrorAnalyzer:
    """Analisis error untuk meningkatkan akurasi"""
    
    def __init__(self):
        self.errors = []
        self.error_patterns = {}
    
    def analyze_transcription_errors(self, reference, hypothesis):
        """Analisis error dalam transkripsi"""
        
        # Split menjadi kata
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Align words (sederhana)
        errors = []
        
        for i, (ref, hyp) in enumerate(zip(ref_words, hyp_words)):
            if ref != hyp:
                error_type = self._classify_error(ref, hyp)
                errors.append({
                    'position': i,
                    'reference': ref,
                    'hypothesis': hyp,
                    'type': error_type
                })
        
        # Analisis pola error
        error_types = Counter([e['type'] for e in errors])
        common_errors = Counter([e['reference'] for e in errors])
        
        return {
            'total_errors': len(errors),
            'error_types': error_types,
            'common_errors': common_errors.most_common(10),
            'wer': len(errors) / len(ref_words),
            'error_details': errors
        }
    
    def _classify_error(self, ref, hyp):
        """Klasifikasi jenis error"""
        
        # Homophone (kata mirip)
        if self._is_homophone(ref, hyp):
            return 'homophone'
        
        # Domain-specific (istilah HR)
        if self._is_hr_term(ref):
            return 'domain_specific'
        
        # Insertion
        if hyp and not ref:
            return 'insertion'
        
        # Deletion
        if ref and not hyp:
            return 'deletion'
        
        # Substitution
        return 'substitution'
    
    def _is_homophone(self, ref, hyp):
        """Deteksi homophone"""
        homophones = {
            'rekrutmen': ['rekruitmen', 'rekrutmen'],
            'payroll': ['payrol', 'payrole'],
            'kinerja': ['kinerja', 'kinerja'],
        }
        
        for key, variants in homophones.items():
            if ref == key and hyp in variants:
                return True
        return False
    
    def _is_hr_term(self, word):
        """Deteksi apakah kata adalah istilah HR"""
        hr_terms = [
            'hrd', 'payroll', 'rekrutmen', 'onboarding', 'kpi',
            'kinerja', 'absensi', 'training', 'development'
        ]
        return word in hr_terms
    
    def generate_improvement_recommendations(self, analysis_result):
        """Generate rekomendasi perbaikan"""
        recommendations = []
        
        # Rekomendasi berdasarkan error types
        if analysis_result['error_types'].get('homophone', 0) > 5:
            recommendations.append({
                'type': 'correction_dict',
                'priority': 'high',
                'suggestion': 'Tambahkan homophone pairs ke correction dictionary',
                'examples': [f"{e['reference']} → {e['hypothesis']}" for e in analysis_result['error_details'][:3]]
            })
        
        if analysis_result['error_types'].get('domain_specific', 0) > 3:
            recommendations.append({
                'type': 'prompt_engineering',
                'priority': 'high',
                'suggestion': 'Perkuat initial prompt dengan istilah HR yang sering salah',
                'examples': [e['reference'] for e in analysis_result['error_details'][:5]]
            })
        
        if analysis_result['error_types'].get('insertion', 0) > 10:
            recommendations.append({
                'type': 'audio_processing',
                'priority': 'medium',
                'suggestion': 'Noise reduction mungkin perlu ditingkatkan',
                'examples': []
            })
        
        return recommendations