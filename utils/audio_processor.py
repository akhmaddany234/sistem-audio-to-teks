from pydub import AudioSegment
import os
import tempfile

def split_audio(audio_path, chunk_duration_ms=300000):  # 5 menit per chunk
    """
    Split audio panjang menjadi chunk-chunk kecil
    """
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    
    for i, start in enumerate(range(0, len(audio), chunk_duration_ms)):
        end = start + chunk_duration_ms
        chunk = audio[start:end]
        
        # Simpan chunk ke temporary file
        temp_chunk = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        chunk.export(temp_chunk.name, format='wav')
        chunks.append(temp_chunk.name)
    
    return chunks

def get_audio_duration(audio_path):
    """
    Dapatkan durasi audio dalam detik
    """
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000  # Konversi ke detik

def reduce_noise(audio_path):
    """
    Sederhana noise reduction (basic)
    """
    audio = AudioSegment.from_file(audio_path)
    
    # Sederhana: reduce noise dengan low-pass filter
    filtered = audio.low_pass_filter(3000)
    
    temp_clean = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    filtered.export(temp_clean.name, format='wav')
    
    return temp_clean.name

def convert_to_wav(audio_path):
    """
    Konversi audio ke format WAV
    """
    audio = AudioSegment.from_file(audio_path)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    audio.export(temp_wav.name, format='wav')
    return temp_wav.name

def get_audio_info(audio_path):
    """
    Dapatkan informasi lengkap audio
    """
    audio = AudioSegment.from_file(audio_path)
    return {
        'duration': len(audio) / 1000,
        'channels': audio.channels,
        'frame_rate': audio.frame_rate,
        'sample_width': audio.sample_width,
        'max_volume': audio.max
    }