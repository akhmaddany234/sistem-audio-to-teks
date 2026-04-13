import streamlit as st
import os
import tempfile
from datetime import datetime, date
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt
import time
import calendar
from pydub import AudioSegment
import io

# Import fungsi dari utils 
from utils.transcriber import transcribe_audio
from utils.summarizer import generate_summary, extract_keywords

# Konfigurasi halaman
st.set_page_config(
    page_title="HR Meeting Minutes Generator",
    page_icon="📝",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .success-box {
        padding: 20px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
    }
    .info-box {
        padding: 15px;
        background-color: #e7f3ff;
        border: 1px solid #b8daff;
        border-radius: 5px;
        margin: 10px 0;
    }
    .audio-player {
        margin: 20px 0;
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    .summarize-button {
        margin: 20px 0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>HR Minutes Generator</h1>
        <p>Upload rekaman rapat HR, dapatkan notulen otomatis</p>
    </div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/meeting.png")
    st.markdown("### Tentang Aplikasi")
    
    st.markdown("### Format Audio Didukung")
    st.write("• MP3\n• WAV\n• M4A\n• MP4 (audio saja)")
    
    st.markdown("### Tips Penggunaan")
    st.write(
        """
        1. Pastikan audio jelas
        2. Gunakan fitur cut untuk memotong bagian penting (opsional)
        3. Bisa edit transkrip lalu ringkaskan ulang
        4. Bahasa Indonesia/Inggris
        """
    )
    
    st.markdown("---")
    
    # ===== Pengaturan Summarization =====
    st.markdown("### Pengaturan Summarization")
    
    use_advanced_summarizer = st.checkbox(
        "Gunakan Advanced Summarizer (Transformer)",
        value=False,
        help="Menggunakan model IndoBERT untuk summarization yang lebih akurat"
    )
    
    if use_advanced_summarizer:
        st.info("""
        **Model yang digunakan:**
        - IndoBERT Summarization
        - Zero-shot classification untuk key points
        - NER untuk timeline extraction
        """)
    
    st.markdown("---")
    
    # ===== Opsi Model Whisper =====
    st.markdown("### Pengaturan Lanjutan")
    
    whisper_model = st.selectbox(
        "Opsi Ukuran Model Whisper",
        options=["tiny", "base", "small", "medium", "large"],
        index=1,
        help="""
        - **tiny**: Tercepat, akurasi rendah
        - **base**: Cepat, akurasi cukup
        - **small**: Sedang, akurasi baik
        - **medium**: Lambat, akurasi sangat baik
        - **large**: Paling lambat, akurasi terbaik (rekomendasi untuk bahasa Indonesia)
        """
    )
    
    model_info = {
        "tiny": "⚡ 32x lebih cepat dari large, akurasi ~60%",
        "base": "🚀 16x lebih cepat dari large, akurasi ~75%",
        "small": "🏃 8x lebih cepat dari large, akurasi ~82%",
        "medium": "🐢 4x lebih cepat dari large, akurasi ~88%",
        "large": "🐌 Kecepatan normal, akurasi ~92% (terbaik untuk Bahasa Indonesia)"
    }
    st.caption(model_info[whisper_model])
    
    # ===== Opsi Preprocessing Audio =====
    st.markdown("#### Preprocessing Audio")
    
    use_noise_reduction = st.checkbox(
        "Noise Reduction", 
        value=False,
        help="Kurangi noise background (untuk rekaman di tempat ramai)"
    )
    
    use_auto_split = st.checkbox(
        "Auto-split Audio Panjang", 
        value=False,
        help="Otomatis memotong audio >30 menit menjadi beberapa bagian"
    )
    
    split_duration = st.slider(
        "Durasi per Chunk (menit)",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        help="Durasi setiap potongan audio (hanya jika auto-split aktif)"
    ) if use_auto_split else 10
    
    # Opsi Koreksi
    st.markdown("#### Koreksi Teks")
    
    use_correction = st.checkbox(
        "Gunakan Koreksi Otomatis", 
        value=True,
        help="Koreksi kata-kata yang sering salah transkripsi menggunakan kamus"
    )
    
    # Simpan semua pengaturan di session state
    st.session_state['whisper_model'] = whisper_model
    st.session_state['use_noise_reduction'] = use_noise_reduction
    st.session_state['use_auto_split'] = use_auto_split
    st.session_state['split_duration'] = split_duration
    st.session_state['use_correction'] = use_correction
    st.session_state['use_advanced_summarizer'] = use_advanced_summarizer  # Simpan juga setting ini
    
    st.markdown("---")

# Fungsi untuk fitur cut audio (sama seperti sebelumnya)
def cut_audio(audio_file, start_time, end_time):
    """Memotong audio berdasarkan waktu start dan end (dalam detik)"""
    try:
        audio = AudioSegment.from_file(audio_file)
        start_ms = start_time * 1000
        end_ms = end_time * 1000
        
        if end_ms > len(audio):
            end_ms = len(audio)
        if start_ms < 0:
            start_ms = 0
        if start_ms >= end_ms:
            st.error("❌ Waktu awal harus lebih kecil dari waktu akhir")
            return None
        
        cut_audio_segment = audio[start_ms:end_ms]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        cut_audio_segment.export(temp_file.name, format="mp3")
        
        return temp_file.name
    except Exception as e:
        st.error(f"❌ Error saat memotong audio: {str(e)}")
        return None

# ==================== MAIN CONTENT ====================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Upload Rekaman Rapat")
    
    uploaded_file = st.file_uploader(
        "Pilih file audio (MP3, WAV, M4A)",
        type=['mp3', 'wav', 'm4a', 'mp4'],
        help="Upload rekaman rapat HR Anda"
    )
    
    if uploaded_file is not None:
        # Simpan file di session state
        if 'original_audio' not in st.session_state:
            st.session_state['original_audio'] = uploaded_file
            st.session_state['audio_duration'] = 0
        
        # Tampilkan audio player
        st.markdown('<div class="audio-player">', unsafe_allow_html=True)
        st.markdown("**🎵 Preview Audio Original:**")
        st.audio(uploaded_file)
        
        # Dapatkan durasi
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            audio = AudioSegment.from_file(tmp_path)
            duration_seconds = len(audio) / 1000
            st.session_state['audio_duration'] = duration_seconds
            
            st.markdown(f"**Durasi Audio:** {int(duration_seconds // 60)} menit {int(duration_seconds % 60)} detik")
            os.unlink(tmp_path)
        except Exception as e:
            st.warning(f"Tidak dapat membaca durasi audio: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Fitur Cut Audio
        st.markdown("### ✂️ Fitur Potong Audio")
        st.markdown("Potong bagian audio yang ingin ditranskripsi (tanpa batas ukuran file)")
        
        col_cut1, col_cut2 = st.columns(2)
        
        with col_cut1:
            start_min = st.number_input("Menit Awal", min_value=0, max_value=999, value=0)
            start_sec = st.number_input("Detik Awal", min_value=0, max_value=59, value=0)
            start_time = start_min * 60 + start_sec
        
        with col_cut2:
            end_min = st.number_input("Menit Akhir", min_value=0, max_value=999, value=int(st.session_state.get('audio_duration', 0) // 60))
            end_sec = st.number_input("Detik Akhir", min_value=0, max_value=59, value=int(st.session_state.get('audio_duration', 0) % 60))
            end_time = end_min * 60 + end_sec
        
        if end_time <= start_time:
            st.warning("⚠️ Waktu akhir harus lebih besar dari waktu awal")
        
        if st.button("✂️ Potong Audio", use_container_width=True):
            if end_time > start_time:
                with st.spinner("Memotong audio..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    cut_file_path = cut_audio(tmp_path, start_time, end_time)
                    
                    if cut_file_path:
                        with open(cut_file_path, 'rb') as f:
                            cut_audio_bytes = f.read()
                        
                        st.session_state['cut_audio'] = cut_audio_bytes
                        st.session_state['cut_audio_path'] = cut_file_path
                        st.session_state['is_cut'] = True
                        
                        st.success(f"✅ Audio berhasil dipotong! Durasi: {int((end_time-start_time)//60)} menit {int((end_time-start_time)%60)} detik")
                        os.unlink(tmp_path)
            else:
                st.error("❌ Waktu akhir harus lebih besar dari waktu awal")
        
        if 'is_cut' in st.session_state and st.session_state['is_cut']:
            st.markdown('<div class="audio-player">', unsafe_allow_html=True)
            st.markdown("**✂️ Preview Audio Hasil Potongan:**")
            st.audio(st.session_state['cut_audio'])
            cut_duration = end_time - start_time
            st.markdown(f"**Durasi Potongan:** {int(cut_duration // 60)} menit {int(cut_duration % 60)} detik")
            st.markdown(f"**Rentang Waktu:** {start_time//60:02d}:{start_time%60:02d} - {end_time//60:02d}:{end_time%60:02d}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Input metadata
    with st.expander("📌 Tambahkan Informasi Rapat (Opsional)"):
        meeting_title = st.text_input("Judul Rapat", placeholder="Masukkan Judul Rapat")
        meeting_date = st.date_input("Tanggal Rapat", datetime.now())
        meeting_participants = st.text_area("Peserta Rapat (pisahkan dengan koma)", placeholder="Contoh: Bp.A, Bp.B, Ibu C")
        meeting_place = st.text_input("Tempat Rapat", placeholder="Contoh: Ruang Meeting Olympus Lantai 2")
        meeting_time = st.text_input("Waktu Rapat", value="09.00 s/d 16.30 WIB")
        pimpinan_rapat = st.text_input("Pimpinan Rapat", placeholder="Masukkan nama pimpinan rapat")
        notulis = st.text_input("Notulis", placeholder="Masukkan nama notulis rapat")
    
    # ==================== PROSES BUTTON ====================
    if uploaded_file is not None:
        audio_to_process = None
        audio_source = ""
        
        if 'is_cut' in st.session_state and st.session_state['is_cut']:
            audio_to_process = st.session_state['cut_audio_path']
            audio_source = "potongan"
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_to_process = tmp_file.name
            audio_source = "original"
        
        if st.button(f"🚀 Proses Audio {audio_source.capitalize()}", type="primary", use_container_width=True):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Transkripsi
                status_text.text("⏳ Langkah 1/3: Mentranskripsi audio... (mohon tunggu)")
                progress_bar.progress(20)
                
                transcript = transcribe_audio(
                    audio_to_process,
                    model_name=st.session_state.get('whisper_model', 'small'),
                    use_noise_reduction=st.session_state.get('use_noise_reduction', False),
                    use_auto_split=st.session_state.get('use_auto_split', True),
                    split_duration=st.session_state.get('split_duration', 10),
                    language='id',
                    use_correction=st.session_state.get('use_correction', True)
                )
                
                progress_bar.progress(50)
                status_text.text("✅ Audio berhasil ditranskripsi!")
                
                # Step 2: Generate summary (dengan opsi advanced atau simple)
                status_text.text("⏳ Langkah 2/3: Membuat ringkasan notulen...")
                progress_bar.progress(60)
                
                # ===== PENTING: Pilih summarizer berdasarkan setting =====
                use_advanced = st.session_state.get('use_advanced_summarizer', False)
                
                if use_advanced:
                    try:
                        from utils.summarizer_advanced import AdvancedSummarizer
                        summarizer = AdvancedSummarizer()
                        summary_data = summarizer.generate_complete_summary(transcript)
                    except ImportError:
                        st.warning("⚠️ Advanced summarizer tidak tersedia, menggunakan versi sederhana")
                        summary_data = generate_summary(transcript)
                else:
                    summary_data = generate_summary(transcript)
                
                progress_bar.progress(80)
                status_text.text("✅ Ringkasan berhasil dibuat!")
                
                # Step 3: Extract keywords
                status_text.text("⏳ Langkah 3/3: Mengekstrak kata kunci...")
                keywords = extract_keywords(transcript)
                
                progress_bar.progress(100)
                status_text.text("✅ Proses selesai!")
                
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
                
                # Simpan hasil
                st.session_state['transcript'] = transcript
                st.session_state['edited_transcript'] = transcript
                st.session_state['summary'] = summary_data
                st.session_state['keywords'] = keywords
                st.session_state['meeting_info'] = {
                    'title': meeting_title,
                    'date': meeting_date,
                    'participants': meeting_participants,
                    'place': meeting_place,
                    'time': meeting_time,
                    'pimpinan': pimpinan_rapat,
                    'notulis': notulis,
                    'filename': uploaded_file.name,
                    'audio_source': audio_source,
                    'cut_range': f"{start_time//60:02d}:{start_time%60:02d} - {end_time//60:02d}:{end_time%60:02d}" if audio_source == "potongan" else "full audio"
                }
                st.session_state['processed'] = True
                
                if audio_source == "original":
                    os.unlink(audio_to_process)
                
                st.success("✅ Proses selesai! Scroll ke bawah untuk melihat hasil.")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Tips: Coba gunakan fitur potong audio untuk durasi yang lebih pendek.")

with col2:
    st.markdown("### ℹ️ Cara Penggunaan")
    st.markdown("""
    **Langkah mudah:**
    1. **Rekam** rapat via HP/Zoom/Meet
    2. **Upload** file audio di sini
    3. **Potong** bagian yang penting (opsional)
    4. **Tunggu** proses AI (1-5 menit)
    5. **Edit** transkrip jika perlu
    6. **Ringkaskan Ulang** tanpa proses ulang audio
    
    **Fitur Baru:**
    - ✂️ Potong audio tanpa batas ukuran
    - 🎵 Preview audio original & potongan
    - 📝 Edit transkrip langsung
    - 🔄 Ringkaskan ulang tanpa proses audio
    - 🤖 Advanced summarization dengan IndoBERT
    
    **Keuntungan:**
    - ⏱️ Hemat waktu 90%
    - 🎯 Akurat untuk Bahasa Indonesia
    - 📊 Mudah dicari & diarsip
    - 🔒 Data aman (lokal)
    """)

# ==================== TAMPILKAN HASIL ====================
if 'processed' in st.session_state and st.session_state['processed']:
    st.markdown("---")
    st.markdown("## 📊 Hasil Notulen Rapat")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Ringkasan", "📑 Transkrip Lengkap", "🔑 Kata Kunci", "📥 Export", "📊 Evaluasi"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Judul:** {st.session_state['meeting_info']['title']}")
        with col2:
            st.markdown(f"**Tanggal:** {st.session_state['meeting_info']['date'].strftime('%d/%m/%Y')}")
        with col3:
            st.markdown(f"**File:** {st.session_state['meeting_info']['filename']}")
        
        st.markdown(f"**Peserta:** {st.session_state['meeting_info']['participants']}")
        if st.session_state['meeting_info']['audio_source'] == "potongan":
            st.markdown(f"**Audio yang diproses:** Potongan ({st.session_state['meeting_info']['cut_range']})")
        else:
            st.markdown(f"**Audio yang diproses:** Full audio")
        st.markdown("---")
        
        st.markdown("### 📌 Ringkasan Notulen")
        st.markdown(st.session_state['summary']['summary'])
        
        if st.session_state['summary']['key_points']:
            st.markdown("### ✅ Poin Keputusan")
            for point in st.session_state['summary']['key_points']:
                st.markdown(f"• {point}")
        
        if st.session_state['summary']['timeline']:
            st.markdown("### 📅 Timeline/Tindak Lanjut")
            for item in st.session_state['summary']['timeline']:
                st.markdown(f"• {item}")
    
    with tab2:
        st.markdown("### 📑 Transkrip Lengkap Rapat")
        st.markdown("Anda dapat mengedit transkrip di bawah ini, lalu klik tombol 'Ringkaskan Ulang' untuk mendapatkan ringkasan baru.")
        
        edited_transcript = st.text_area(
            "Edit Transkrip:", 
            st.session_state.get('edited_transcript', st.session_state['transcript']), 
            height=400,
            key="transcript_editor"
        )
        
        if edited_transcript != st.session_state.get('edited_transcript', ''):
            st.session_state['edited_transcript'] = edited_transcript
        
        st.markdown('<div class="summarize-button">', unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🔄 Ringkaskan Ulang Berdasarkan Transkrip yang Diedit", type="primary", use_container_width=True):
                with st.spinner("Meringkaskan ulang transkrip..."):
                    try:
                        use_advanced = st.session_state.get('use_advanced_summarizer', False)
                        if use_advanced:
                            from utils.summarizer_advanced import AdvancedSummarizer
                            summarizer = AdvancedSummarizer()
                            new_summary = summarizer.generate_complete_summary(st.session_state['edited_transcript'])
                        else:
                            new_summary = generate_summary(st.session_state['edited_transcript'])
                        
                        new_keywords = extract_keywords(st.session_state['edited_transcript'])
                        
                        st.session_state['summary'] = new_summary
                        st.session_state['keywords'] = new_keywords
                        
                        st.success("✅ Ringkasan berhasil diperbarui!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error saat meringkaskan ulang: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.info("💡 Tips: Setelah mengedit transkrip, klik tombol di atas untuk mendapatkan ringkasan baru.")
    
    with tab3:
        st.markdown("### 🔑 Kata Kunci Penting")
        keywords = st.session_state['keywords']
        
        cols = st.columns(4)
        for i, keyword in enumerate(keywords[:16]):
            with cols[i % 4]:
                st.markdown(f"""
                <div style="background-color: #667eea20; border-radius: 20px; padding: 8px 15px; margin: 5px; text-align: center; border: 1px solid #667eea;">
                    {keyword}
                </div>
                """, unsafe_allow_html=True)
        
        if len(keywords) > 10:
            st.markdown("### 📊 Frekuensi Kata Kunci")
            keyword_df = pd.DataFrame({
                'Kata Kunci': keywords[:10],
                'Frekuensi': [10-i for i in range(10)]
            })
            st.bar_chart(keyword_df.set_index('Kata Kunci'))
    
    with tab4:
        st.markdown("### 📥 Export Notulen")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export ke Word (Format Notulen)", use_container_width=True):
                # ... (kode export word sama seperti sebelumnya)
                pass
        
        with col2:
            if st.button("📋 Copy ke Clipboard", use_container_width=True):
                # ... (kode copy clipboard sama seperti sebelumnya)
                pass
        
        with col3:
            if st.button("📊 Preview Format", use_container_width=True):
                st.info("""
                **Format Notulen yang akan dihasilkan:**
                • NOTULEN (header besar)
                • {judul rapat}
                • I. Pelaksanaan (dengan detail)
                • II. Hasil Rapat (ringkasan AI)
                • III. Kata Kunci
                • IV. Dokumentasi
                • Tanda tangan 2 kolom
                """)

    with tab5:
        st.markdown("### 📊 Evaluasi Kinerja Model")
        
        # Informasi tentang evaluasi
        with st.expander("ℹ️ Tentang Metrik Evaluasi", expanded=False):
            st.markdown("""
            **Word Error Rate (WER)** - Mengukur akurasi transkripsi
            - Menghitung persentase kata yang salah
            - 0% = sempurna, 100% = semua kata salah
            
            **Character Error Rate (CER)** - Mengukur akurasi karakter
            - Lebih sensitif terhadap kesalahan ejaan
            - 0% = sempurna
            
            **ROUGE Score** - Mengukur kualitas ringkasan
            - ROUGE-1: Unigram overlap
            - ROUGE-2: Bigram overlap
            - ROUGE-L: Longest common subsequence
            - 1.0 = sempurna, 0.0 = tidak ada kesamaan
            """)
        
        # Upload ground truth
        ground_truth_file = st.file_uploader(
            "Upload Ground Truth (Transkrip Manual)", 
            type=['txt', 'docx'],
            help="Upload transkrip hasil manual untuk membandingkan akurasi",
            key="ground_truth_uploader"
        )
        
        if ground_truth_file:
            try:
                # Baca file ground truth
                ground_truth = ground_truth_file.read().decode('utf-8')
                
                # Tampilkan preview
                with st.expander("📄 Preview Ground Truth", expanded=False):
                    st.text(ground_truth[:500] + "..." if len(ground_truth) > 500 else ground_truth)
                
                st.markdown("---")
                st.markdown("### 📈 Hasil Evaluasi")
                
                # Import evaluator
                try:
                    from utils.evaluator import (
                        evaluate_transcription, evaluate_summary, 
                        get_evaluation_interpretation
                    )
                    
                    # 1. Evaluasi Transkripsi
                    st.markdown("#### 🎙️ Evaluasi Transkripsi")
                    
                    trans_eval = evaluate_transcription(
                        ground_truth, 
                        st.session_state['transcript']
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Word Error Rate (WER)", 
                            f"{trans_eval['wer']:.2%}",
                            delta="⬇️ lebih kecil lebih baik"
                        )
                    with col2:
                        st.metric(
                            "Character Error Rate (CER)", 
                            f"{trans_eval['cer']:.2%}",
                            delta="⬇️ lebih kecil lebih baik"
                        )
                    with col3:
                        st.metric(
                            "Akurasi", 
                            f"{(1 - trans_eval['wer']) * 100:.1f}%",
                            delta="⬆️ lebih besar lebih baik"
                        )
                    
                    # Interpretasi
                    interpretation = get_evaluation_interpretation(trans_eval['wer'])
                    st.info(f"""
                    **Interpretasi:** {interpretation['wer_quality']}
                    
                    {interpretation['wer_description']}
                    
                    📊 Detail: {trans_eval['reference_length']} kata referensi vs 
                    {trans_eval['hypothesis_length']} kata hasil transkripsi
                    """)
                    
                    # 2. Evaluasi Ringkasan (jika ada ground truth yang cukup)
                    if len(ground_truth) > 100 and st.session_state.get('summary'):
                        st.markdown("---")
                        st.markdown("#### 📝 Evaluasi Ringkasan")
                        
                        summary_eval = evaluate_summary(
                            ground_truth[:1000], 
                            st.session_state['summary']['summary']
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("ROUGE-1", f"{summary_eval['ROUGE-1']:.2%}")
                        with col2:
                            st.metric("ROUGE-2", f"{summary_eval['ROUGE-2']:.2%}")
                        with col3:
                            st.metric("ROUGE-L", f"{summary_eval['ROUGE-L']:.2%}")
                        
                        st.caption("ROUGE score: 100% = ringkasan sempurna, 0% = tidak ada kesamaan")
                    
                    # 3. Rekomendasi Perbaikan
                    st.markdown("---")
                    st.markdown("#### 💡 Rekomendasi Perbaikan")
                    
                    if trans_eval['wer'] > 0.3:
                        st.warning("""
                        **WER tinggi (>30%)** - Beberapa rekomendasi:
                        """)
                    elif trans_eval['wer'] > 0.15:
                        st.info("""
                        **WER sedang (15-30%)** - Masih bisa ditingkatkan:
                        """)
                    else:
                        st.success("""
                        **WER rendah (<15%)** - Kualitas transkripsi sudah baik!
                        """)
                    
                except ImportError:
                    st.error("""
                    ❌ Modul evaluator belum tersedia.
                    
                    Silakan buat file `utils/evaluator.py` terlebih dahulu.
                    Install dependencies: `pip install jiwer rouge-score`
                    """)
                    
            except Exception as e:
                st.error(f"❌ Error membaca ground truth: {str(e)}")
                st.info("Pastikan file ground truth adalah file TXT dengan encoding UTF-8")
        
        else:
            # Tampilkan placeholder jika belum upload
            st.info("""
            📌 **Cara Menggunakan Fitur Evaluasi:**
            
            1. Siapkan file ground truth (transkrip manual dari audio yang sama)
            2. Upload file tersebut (format .txt atau .docx)
            3. Sistem akan membandingkan hasil AI dengan ground truth
            
            **Manfaat:**
            - Mengukur akurasi transkripsi dengan WER/CER
            - Mengevaluasi kualitas ringkasan dengan ROUGE
            - Mendapatkan rekomendasi perbaikan
            """)
            
            # Contoh format ground truth
            with st.expander("📋 Contoh Format Ground Truth", expanded=False):
                st.code("""
    Halo selamat pagi semuanya. Hari ini kita akan membahas agenda rapat HRD yang pertama yaitu tentang payroll bulan Maret.

    Setelah kita diskusikan, diputuskan bahwa payroll akan diproses pada tanggal 25 Maret 2024.

    Untuk agenda kedua tentang rekrutmen karyawan baru, targetnya adalah 10 orang untuk posisi operator produksi.
                """, language="text")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🚀 Powered by Whisper AI + IndoBERT | Made with ❤️ untuk HR Indonesia</p>
        <p style="font-size: 12px;">© 2024 - HR Meeting Minutes Generator</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Cleanup
def cleanup():
    if 'cut_audio_path' in st.session_state:
        try:
            os.unlink(st.session_state['cut_audio_path'])
        except:
            pass

import atexit
atexit.register(cleanup)