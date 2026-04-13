import pandas as pd
import streamlit as st
import re

@st.cache_data
def load_corrections(csv_path="utils/correction_kamus.csv"):
    """
    Load kamus koreksi dari file CSV
    Format CSV: wrong,correct
    """
    df = pd.read_csv(csv_path)
    return dict(zip(df["wrong"], df["correct"]))

def correct_transcript(text, csv_path="utils/correction_kamus.csv"):
    """
    Koreksi seluruh teks transkrip menggunakan kamus
    """
    corrections = load_corrections(csv_path)
    corrected = text

    # Urutkan dari kata terpanjang dulu (hindari partial replacement)
    sorted_corrections = sorted(
        corrections.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    for wrong, correct in sorted_corrections:
        # Gunakan word boundary agar tidak salah koreksi di tengah kata
        pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)
        corrected = pattern.sub(correct, corrected)

    return corrected

def correct_keywords(keywords, csv_path="utils/correction_kamus.csv"):
    """
    Koreksi daftar keywords
    """
    corrections = load_corrections(csv_path)
    
    # Buat mapping lowercase untuk memudahkan pencarian
    corrections_lower = {k.lower(): v for k, v in corrections.items()}
    
    corrected = []
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in corrections_lower:
            corrected.append(corrections_lower[keyword_lower])
        else:
            corrected.append(keyword)
    
    return corrected

# Fungsi tambahan untuk menambah koreksi baru (opsional)
def add_correction(wrong, correct, csv_path="utils/correction_kamus.csv"):
    """
    Tambah koreksi baru ke file CSV
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["wrong", "correct"])
    
    # Cek apakah sudah ada
    if wrong in df["wrong"].values:
        # Update yang existing
        df.loc[df["wrong"] == wrong, "correct"] = correct
    else:
        # Tambah baru
        new_row = pd.DataFrame({"wrong": [wrong], "correct": [correct]})
        df = pd.concat([df, new_row], ignore_index=True)
    
    df.to_csv(csv_path, index=False)
    st.cache_data.clear()  # Clear cache agar reload data baru