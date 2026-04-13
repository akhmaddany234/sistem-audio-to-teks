# utils/transcriber.py
import whisper
import os
from pydub import AudioSegment
import tempfile
import streamlit as st
from .corrections import correct_transcript
from .audio_processor import split_audio, get_audio_duration, reduce_noise  # Import dari audio_processor

# Cache model berdasarkan nama model
@st.cache_resource
def load_whisper_model(model_name="small"):
    """Load Whisper model (cached) berdasarkan pilihan user"""
    with st.spinner(f"🔄 Memuat model Whisper {model_name}..."):
        return whisper.load_model(model_name)

def convert_audio_to_wav(input_path):
    """Convert audio to WAV format if needed"""
    try:
        # Cek ekstensi file
        file_ext = os.path.splitext(input_path)[1].lower()
        
        if file_ext == '.wav':
            return input_path
        
        # Konversi ke wav
        audio = AudioSegment.from_file(input_path)
        
        # Buat temporary file untuk wav
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        # Export ke wav
        audio.export(temp_wav_path, format='wav')
        
        return temp_wav_path
    except Exception as e:
        raise Exception(f"Error converting audio: {str(e)}")

def transcribe_audio(audio_path, 
                    model_name="small",
                    use_noise_reduction=False,
                    use_auto_split=True,
                    split_duration=10,
                    language='id',
                    use_correction=True):
    """
    Transkripsi audio menggunakan Whisper dengan berbagai opsi
    
    Parameters:
    - audio_path: path ke file audio
    - model_name: tiny/base/small/medium/large
    - use_noise_reduction: boolean, apakah pakai noise reduction
    - use_auto_split: boolean, apakah auto-split untuk audio panjang
    - split_duration: durasi per chunk dalam menit
    - language: kode bahasa (id/en)
    - use_correction: boolean, apakah pakai koreksi otomatis
    """
    try:
        # Load model sesuai pilihan
        model = load_whisper_model(model_name)
        
        # Step 1: Konversi ke WAV
        current_audio = convert_audio_to_wav(audio_path)
        
        # Step 2: Noise reduction (jika dipilih)
        if use_noise_reduction:
            st.info("🔊 Menerapkan noise reduction...")
            current_audio = reduce_noise(current_audio)
        
        # Step 3: Cek durasi untuk informasi
        duration = get_audio_duration(current_audio)
        st.info(f"📊 Durasi audio: {int(duration // 60)} menit {int(duration % 60)} detik")
        
        # Step 4: Split audio jika panjang dan auto-split aktif
        if use_auto_split and duration > 30 * 60:  # > 30 menit
            st.info(f"✂️ Audio panjang ({int(duration//60)} menit), akan diproses per bagian...")
            
            # Split audio (konversi menit ke milidetik)
            chunks = split_audio(current_audio, chunk_duration_ms=split_duration * 60 * 1000)
            
            transcripts = []
            total_chunks = len(chunks)
            
            # Progress bar untuk chunks
            chunk_progress = st.progress(0)
            chunk_status = st.empty()
            
            for i, chunk_path in enumerate(chunks):
                chunk_status.text(f"📝 Memproses bagian {i+1}/{total_chunks}...")
                
                # Transkrip per chunk
                result = model.transcribe(
                    chunk_path,
                    language=language,
                    task='transcribe',
                    fp16=False,
                    initial_prompt="Rapat HR membahas payroll, onboarding, KPI, evaluasi kinerja, rekrutmen, training, deadline.",
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=True
                )
                
                transcripts.append(result['text'])
                
                # Update progress
                chunk_progress.progress((i + 1) / total_chunks)
                
                # Hapus file chunk
                try:
                    os.unlink(chunk_path)
                except:
                    pass
            
            chunk_status.empty()
            chunk_progress.empty()
            
            # Gabungkan semua transkrip
            full_transcript = ' '.join(transcripts)
            
        else:
            # Transkrip langsung tanpa split
            result = model.transcribe(
                current_audio,
                language=language,
                task='transcribe',
                fp16=False,
                initial_prompt="Rapat HR membahas payroll, onboarding, KPI, evaluasi kinerja, rekrutmen, training, deadline.",
                temperature=0.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True
            )
            
            full_transcript = result['text']
        
        # Step 5: Terapkan koreksi jika dipilih
        if use_correction:
            try:
                corrected_transcript = correct_transcript(full_transcript)
            except Exception as e:
                st.warning(f"Koreksi otomatis gagal: {str(e)}")
                corrected_transcript = full_transcript
        else:
            corrected_transcript = full_transcript
        
        # Bersihkan file temporary
        if current_audio != audio_path and os.path.exists(current_audio):
            try:
                os.unlink(current_audio)
            except:
                pass
        
        return corrected_transcript
    
    except Exception as e:
        # Fallback sederhana
        try:
            st.warning(f"Error dengan parameter lanjutan, mencoba transkripsi dasar...")
            model = whisper.load_model("base")
            result = model.transcribe(audio_path, language='id')
            return result['text']
        except:
            raise Exception(f"Transcription failed: {str(e)}")

# Fungsi wrapper untuk compatibility dengan kode lama
def transcribe_audio_simple(audio_path):
    """Versi sederhana untuk backward compatibility"""
    return transcribe_audio(
        audio_path,
        model_name="small",
        use_noise_reduction=False,
        use_auto_split=True,
        use_correction=True
    )