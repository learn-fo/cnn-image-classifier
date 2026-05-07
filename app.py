import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🧠",
    layout="centered"
)

# ============================================
# NAMA KELAS CIFAR-10
# ============================================
CLASSES = [
    '✈️ Pesawat', '🚗 Mobil', '🐦 Burung', '🐱 Kucing', '🦌 Rusa',
    '🐕 Anjing', '🐸 Katak', '🐴 Kuda', '🚢 Kapal', '🚛 Truk'
]

# ============================================
# MEMBANGUN MODEL CNN
# ============================================
@st.cache_resource
def load_model():
    """Membangun model CNN dari scratch"""
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================
# PREPROCESS GAMBAR
# ============================================
def preprocess_image(image):
    """Mengubah gambar menjadi format yang bisa diproses model"""
    image = image.resize((32, 32))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ============================================
# UI UTAMA
# ============================================
st.title("🧠 CNN Image Classifier")
st.markdown("*Upload gambar untuk klasifikasi ke 10 kategori CIFAR-10*")
st.divider()

# Load model (belum dilatih, akan random)
model = load_model()
st.info("ℹ️ Model menggunakan bobot random. Untuk hasil akurat, model perlu dilatih terlebih dahulu.")

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("🎯 10 Kategori")
    for i, c in enumerate(CLASSES):
        st.write(f"{i}. {c}")
    
    st.divider()
    
    st.header("📌 Cara Pakai")
    st.markdown("""
    1. Upload gambar (JPG/PNG)
    2. Klik tombol Klasifikasi
    3. Lihat hasil prediksi
    """)

# ============================================
# UPLOAD GAMBAR
# ============================================
uploaded_file = st.file_uploader(
    "📤 Pilih gambar",
    type=['jpg', 'jpeg', 'png'],
    help="Upload gambar dengan format JPG, JPEG, atau PNG"
)

if uploaded_file is not None:
    # Buka dan tampilkan gambar
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar yang diunggah", width=250)
    
    # Tombol klasifikasi
    if st.button("🔍 KLASIFIKASI", type="primary", use_container_width=True):
        with st.spinner("🧠 Memproses gambar..."):
            processed = preprocess_image(image)
            predictions = model.predict(processed, verbose=0)[0]
            class_id = np.argmax(predictions)
            confidence = predictions[class_id] * 100
        
        # Tampilkan hasil
        st.markdown("---")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; 
                    border-radius: 15px; 
                    text-align: center;
                    color: white;">
            <p style="font-size: 0.9rem; opacity: 0.8;">HASIL KLASIFIKASI</p>
            <h1 style="font-size: 2.5rem; margin: 0;">{CLASSES[class_id]}</h1>
            <p style="font-size: 1.2rem;">Keyakinan: <b>{confidence:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(confidence / 100)

st.divider()
st.caption("🧠 CNN Image Classifier | Dibangun dengan TensorFlow & Streamlit")
