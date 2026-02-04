import streamlit as st
import numpy as np
import time
from PIL import Image, ImageOps
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# --- ADVANCED UI CONFIG ---
st.set_page_config(page_title="NEURAL DIGIT SCANNER", layout="centered")

# --- CUSTOM CREATIVE CSS WITH BACKGROUND IMAGES ---
st.markdown("""
    <style>
    /* 1. Kinetic Neural Background */
    .stApp {
        background: 
            linear-gradient(rgba(2, 6, 23, 0.85), rgba(2, 6, 23, 0.85)),
            url("https://www.transparenttextures.com/patterns/carbon-fibre.png"),
            url("https://images.unsplash.com/photo-1635070041078-e363dbe005cb?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
        color: #f8fafc;
    }

    /* 3. Title with Glow & Image Icon */
    .main-title {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(to right, #818cf8, #fb7185);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4rem !important;
        font-weight: 900 !important;
        text-align: center;
        margin-bottom: 0px;
        filter: drop-shadow(0 0 15px rgba(129, 140, 248, 0.4));
    }

    /* 4. Result Glow Box */
    .prediction-box {
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid #818cf8;
        box-shadow: 0 0 30px rgba(129, 140, 248, 0.3);
        padding: 40px;
        border-radius: 30px;
        text-align: center;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 20px rgba(129, 140, 248, 0.3); }
        50% { box-shadow: 0 0 50px rgba(251, 113, 133, 0.4); }
        100% { box-shadow: 0 0 20px rgba(129, 140, 248, 0.3); }
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA & MODEL (UNCHANGED) ---
digits = load_digits()
X_train, _, y_train, _ = train_test_split(digits.data, digits.target, train_size=0.7, random_state=42)
model = SVC(kernel="rbf", gamma="scale", C=1.0)
model.fit(X_train, y_train)

# --- UI LAYOUT ---
st.markdown("<h1 class='main-title'>NEURAL SCANNER</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8; letter-spacing: 5px; font-size: 0.8rem; margin-bottom: 50px;'>PROBABILISTIC DIGIT CLASSIFICATION v2.0</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with st.spinner("⚡ DECODING PIXEL VECTORS..."):
        time.sleep(1.2)
    
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    # Preprocessing
    proc_img = ImageOps.grayscale(image).resize((8, 8), Image.BILINEAR)
    proc_img = ImageOps.pad(proc_img, (8, 8), color=255)
    img_array = np.array(proc_img) / 255.0 * 16.0
    if img_array.mean() > 8: img_array = 16 - img_array
    
    with col1:
        st.markdown("<p style='text-align:center; color:#818cf8;'>USER INPUT</p>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    with col2:
        st.markdown("<p style='text-align:center; color:#fb7185;'>AI PERCEPTION</p>", unsafe_allow_html=True)
        st.image(img_array / 16.0, use_container_width=True)

    # Prediction
    pred = model.predict(img_array.reshape(1, -1))[0]

    st.markdown(f"""
        <div class="prediction-box">
            <p style="color: #94a3b8; font-size: 1rem; margin-bottom:0;">THE NEURAL VERDICT IS</p>
            <h1 style="font-size: 120px !important; margin:0; color: white;">{pred}</h1>
            <div style="background: #818cf8; height: 2px; width: 50px; margin: 10px auto;"></div>
        </div>
    """, unsafe_allow_html=True)

else:
    # Empty space with a centered icon when no file is uploaded
    st.markdown("<h1 style='text-align:center; opacity:0.1; font-size:150px;'>🧠</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color:#334155; margin-top:100px;'>SECURE TERMINAL | ENCRYPTED INFERENCE ENGINE</p>", unsafe_allow_html=True)