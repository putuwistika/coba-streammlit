import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Fungsi untuk memuat model dengan caching
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "iris_model.joblib"  # Sesuaikan path jika model Anda disimpan di tempat lain
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' tidak ditemukan. Pastikan file model telah diunggah ke proyek Streamlit Cloud Anda.")

# Memuat model
model = load_model()

# Judul aplikasi
st.title("Prediksi Jenis Bunga Iris")

# Form input
st.write("Masukkan nilai fitur bunga iris:")
sepal_length = st.number_input("Panjang sepal", min_value=0.0)
sepal_width = st.number_input("Lebar sepal", min_value=0.0)
petal_length = st.number_input("Panjang petal", min_value=0.0)
petal_width = st.number_input("Lebar petal", min_value=0.0)

# Tombol prediksi
if st.button("Prediksi"):
    # Memastikan input valid
    if all(value >= 0 for value in [sepal_length, sepal_width, petal_length, petal_width]):
        # Membentuk data input
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Melakukan prediksi
        prediction = model.predict(input_data)

        # Menampilkan hasil prediksi
        st.success(f"Prediksi: {['setosa', 'versicolor', 'virginica'][prediction[0]]}")
    else:
        st.error("Nilai input harus berupa bilangan positif.")