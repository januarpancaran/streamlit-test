"""
Untuk menjalankan aplikasi streamlit secara local, lakukan instalasi modul streamlit melalui command prompt dengan perintah
`pip install streamlit`, kemudian setelah berhasil terinstall aplikasi dapat berjalan dengan mengetikkan perintah
`streamlit run app.py` pada tempat dimana kamu menyimpan file app.py milikmu. Jangan lupa tambahkan file requirements juga
yang berisi library python yang dipakai agar aplikasi bisa berjalan.
"""

import os
import time
import urllib.request

import numpy as np
import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image
from tensorflow.keras.models import load_model

st.title("🪨📄✂️ Rock Paper Scissors Classifier")

st.markdown(
    """
Upload gambar tangan membentuk **rock (batu)**, **paper (kertas)**, atau **scissors (gunting)**.  
Aplikasi ini akan mengklasifikasikannya secara otomatis menggunakan model CNN.
"""
)

url = "https://github.com/januarpancaran/streamlit-test/releases/download/test/rps-dicoding.h5"
local_path = "model/rps-dicoding.h5"

# Check & download model
if not os.path.exists(local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        with st.spinner("🔄 Mengunduh model dari GitHub Releases..."):
            urllib.request.urlretrieve(url, local_path)
            st.success("✅ Model berhasil diunduh.")
    except Exception as e:
        st.error(f"❌ Gagal mengunduh model: {e}")
else:
    st.info("📦 Model sudah tersedia secara lokal.")


# Prediction function
def predict(image_file):
    classifier_model = local_path

    model = load_model(classifier_model)

    img = Image.open(image_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    # Tentukan hasil klasifikasi
    class_names = ["PAPER", "ROCK", "SCISSORS"]
    predicted_class = class_names[np.argmax(classes)]
    confidence = 100 * np.max(classes)

    return predicted_class, confidence


# Main app function
def main():
    file_uploaded = st.file_uploader("Pilih gambar...", type=["png", "jpg", "jpeg"])
    if file_uploaded is not None:
        image_display = Image.open(file_uploaded)
        st.image(image_display, caption="Gambar yang diupload")

    if st.button("🔍 Klasifikasi"):
        if file_uploaded is None:
            st.warning("Silakan upload gambar terlebih dahulu.")
        else:
            with st.spinner("Model sedang memproses..."):
                label, confidence = predict(file_uploaded)
                time.sleep(1)
                st.success("Selesai diklasifikasi!")
                st.markdown(f"### Hasil: **{label}**")
                st.markdown(f"**Confidence:** {confidence:.2f}%")


if __name__ == "__main__":
    main()
