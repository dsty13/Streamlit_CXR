import streamlit as st
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib  # Ganti pickle dengan joblib


# Define the paths where the files will be saved locally
scaler_path = 'scaler_best.pkl'
ffnn_model_path = 'ffnn_model_best.h5'


# Load the scaler
scaler = joblib.load(scaler_path)


# Load the FFNN model
ffnn_model = tf.keras.models.load_model(ffnn_model_path)

# Load ResNet50 for feature extraction
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features using ResNet50
def extract_features_from_image(img_path):
    try:
        # Load and preprocess image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        # Extract features using ResNet50
        features = resnet_model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Dictionary for sample images
sample_images = {
    "COVID-19": [
        "images/covid19/covid1.png",
        "images/covid19/covid2.png",
        "images/covid19/covid3.png"
    ],
    "Pneumonia": [
        "images/pneumonia/pneumonia1.png",
        "images/pneumonia/pneumonia2.png",
        "images/pneumonia/pneumonia3.png"
    ],
    "Normal": [
        "images/normal/normal1.png",
        "images/normal/normal2.png",
        "images/normal/normal3.png"
    ]
    }

# CSS untuk styling tombol
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        height: 50px;
        border-radius: 5px;
        font-size: 16px;
    }
    .stButton>button:nth-child(1) {
        background-color: #FF4B4B;
        color: white;
    }
    .stButton>button:nth-child(2) {
        background-color: white;
        color: black;
        border: 2px solid #FF4B4B;
    }
    .stButton>button:hover {
        background-color: #e64444;
    }
    </style>
""", unsafe_allow_html=True)


# Setup session state untuk navigasi
if "page" not in st.session_state:
    st.session_state.page = "home"
 
# Halaman utama
if st.session_state.page == "home":
   # Judul aplikasi dengan garis merah di bawahnya
    st.title("KLASIFIKASI COVID-19 PADA CITRA XRAY DADA")
    st.markdown('<hr style="border: 1px solid red;">', unsafe_allow_html=True)  # Garis merah
    
    # Deskripsi aplikasi
    st.write("1. Aplikasi ini hanya mendukung Citra XRAY Dada")
    st.write("2. Hanya dapat mengklasifikasikan 3 kelas yaitu COVID-19, Pneumonia, Normal")
    st.write("3. Ekstraksi fitur menggunakan ResNet50 dan menggunakan FeedForward Neural Network untuk Klasifikasi")
    st.write("4. Akurasi model mencapai 92%")
    st.write("5. Dataset diambil dari Kaggle, berikut jika ingin mengunduh dataset:")

    # Menambahkan tombol untuk mengunduh dataset dan melakukan redirect
    if st.button('Unduh Dataset'):
        # Redirect ke halaman Kaggle
        st.markdown('<a href="https://www.kaggle.com/datasets/anasmohammedtahir/covidqu" target="_blank">Klik di sini untuk mengunduh dataset</a>', unsafe_allow_html=True)
    
    # Tombol dengan Streamlit menggunakan layout kolom
col1, col2 = st.columns([1, 1])  # Membuat dua kolom dengan ukuran yang sama

# Kolom pertama (tombol merah)
with col1:
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    if st.button("Input Image", key="input_image"):
        st.session_state.page = "input_image"
    st.markdown('</div>', unsafe_allow_html=True)

# Kolom kedua (tombol putih)
with col2:
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    if st.button("Try Sample", key="try_sample"):
        st.session_state.page = "try_sample"
    st.markdown('</div>', unsafe_allow_html=True)

# Halaman input gambar
if st.session_state.page == "input_image":
    st.title("Input Gambar")
    uploaded_file = st.file_uploader("Unggah gambar", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image("temp_image.png", caption="Gambar yang diunggah", use_container_width=True)

        if st.button("Prediksi"):
            features = extract_features_from_image("temp_image.png")
            if features is not None:
                scaled_features = scaler.transform([features])
                prediction = ffnn_model.predict(scaled_features)
                predicted_class = np.argmax(prediction, axis=1)

                if predicted_class == 0:
                    st.write("Prediksi: **COVID-19**")
                elif predicted_class == 1:
                    st.write("Prediksi: **Pneumonia**")
                elif predicted_class == 2:
                    st.write("Prediksi: **Normal**")

    if st.button("Kembali"):
        st.session_state.page = "home"


# Halaman gambar sampel
if st.session_state.page == "try_sample":
    st.title("Coba Gambar Sampel")

    # Membuat 3 kolom untuk menampilkan gambar sampel sesuai dengan kelasnya
    col1, col2, col3 = st.columns(3)  # Membuat tiga kolom

    # Kolom pertama untuk COVID-19
    with col1:
        st.subheader("COVID-19")
        for image_path in sample_images["COVID-19"]:
            filename = image_path.split('/')[-1]  # Mendapatkan nama file
            if st.button(f"Prediksi {filename}", key=image_path):  # Menampilkan nama file di tombol
                st.image(image_path, caption=f"Gambar Sampel: {filename}", use_container_width=True)
                features = extract_features_from_image(image_path)
                if features is not None:
                    scaled_features = scaler.transform([features])
                    prediction = ffnn_model.predict(scaled_features)
                    predicted_class = np.argmax(prediction, axis=1)

                    if predicted_class == 0:
                        st.write("Prediksi: **COVID-19**")
                    elif predicted_class == 1:
                        st.write("Prediksi: **Pneumonia**")
                    elif predicted_class == 2:
                        st.write("Prediksi: **Normal**")

    # Kolom kedua untuk Pneumonia
    with col2:
        st.subheader("Pneumonia")
        for image_path in sample_images["Pneumonia"]:
            filename = image_path.split('/')[-1]  # Mendapatkan nama file
            if st.button(f"Prediksi {filename}", key=image_path):  # Menampilkan nama file di tombol
                st.image(image_path, caption=f"Gambar Sampel: {filename}", use_container_width=True)
                features = extract_features_from_image(image_path)
                if features is not None:
                    scaled_features = scaler.transform([features])
                    prediction = ffnn_model.predict(scaled_features)
                    predicted_class = np.argmax(prediction, axis=1)

                    if predicted_class == 0:
                        st.write("Prediksi: **COVID-19**")
                    elif predicted_class == 1:
                        st.write("Prediksi: **Pneumonia**")
                    elif predicted_class == 2:
                        st.write("Prediksi: **Normal**")

    # Kolom ketiga untuk Normal
    with col3:
        st.subheader("Normal")
        for image_path in sample_images["Normal"]:
            filename = image_path.split('/')[-1]  # Mendapatkan nama file
            if st.button(f"Prediksi {filename}", key=image_path):  # Menampilkan nama file di tombol
                st.image(image_path, caption=f"Gambar Sampel: {filename}", use_container_width=True)
                features = extract_features_from_image(image_path)
                if features is not None:
                    scaled_features = scaler.transform([features])
                    prediction = ffnn_model.predict(scaled_features)
                    predicted_class = np.argmax(prediction, axis=1)

                    if predicted_class == 0:
                        st.write("Prediksi: **COVID-19**")
                    elif predicted_class == 1:
                        st.write("Prediksi: **Pneumonia**")
                    elif predicted_class == 2:
                        st.write("Prediksi: **Normal**")

    if st.button("Kembali"):
        st.session_state.page = "home"