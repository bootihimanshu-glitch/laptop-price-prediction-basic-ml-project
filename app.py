import sys
import os

import warnings
warnings.filterwarnings("ignore")

# Fix for scikit-learn CyHalfSquaredError unpickling in modern environments
try:
    import sklearn._loss._loss
    sys.modules['_loss'] = sklearn._loss._loss
except Exception:
    pass

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
)

# Custom CSS for clean, modern look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .price-card {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.4);
    }
    .price-value {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .price-label {
        font-size: 0.95rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model & Dataset ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "pipe.pkl")
    with open(model_path, "rb") as f:
        pipe = pickle.load(f)
    return pipe

@st.cache_resource
def load_dataframe():
    df_path = os.path.join(os.path.dirname(__file__), "sf.pkl")
    with open(df_path, "rb") as f:
        df = pickle.load(f)
    return df

pipe = load_model()
df = load_dataframe()

st.markdown('<div class="main-header">💻 Laptop Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Select laptop specifications to estimate market price based on machine learning</div>', unsafe_allow_html=True)

# --- Input Form ---
col_left, col_right = st.columns(2, gap="medium")

with col_left:
    st.subheader("⚙️ Core Specifications")
    company = st.selectbox("Brand", sorted(df["Company"].unique()), index=sorted(df["Company"].unique()).index("Dell") if "Dell" in df["Company"].unique() else 0)
    type_name = st.selectbox("Laptop Type", sorted(df["TypeName"].unique()), index=sorted(df["TypeName"].unique()).index("Notebook") if "Notebook" in df["TypeName"].unique() else 0)
    ram = st.selectbox("RAM (GB)", sorted(df["Ram"].unique()), index=sorted(df["Ram"].unique()).index(8) if 8 in df["Ram"].unique() else 0)
    cpu_brand = st.selectbox("Processor (CPU)", sorted(df["Cpu_brand"].unique()), index=sorted(df["Cpu_brand"].unique()).index("Intel Core i5") if "Intel Core i5" in df["Cpu_brand"].unique() else 0)
    gpu_brand = st.selectbox("GPU Brand", sorted(df["brand_gpu"].unique()), index=sorted(df["brand_gpu"].unique()).index("Intel") if "Intel" in df["brand_gpu"].unique() else 0)
    os_type = st.selectbox("Operating System", sorted(df["os"].unique()), index=sorted(df["os"].unique()).index("Windows") if "Windows" in df["os"].unique() else 0)

with col_right:
    st.subheader("🖥️ Display & Storage")
    
    col_screen1, col_screen2 = st.columns(2)
    with col_screen1:
        screen_size = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
    with col_screen2:
        resolution = st.selectbox(
            "Screen Resolution",
            [
                "1920x1080 (Full HD)",
                "1366x768 (HD)",
                "1600x900 (HD+)",
                "2560x1440 (2K)",
                "2560x1600 (Retina / QHD+)",
                "2880x1800 (Retina)",
                "3840x2160 (4K UHD)",
            ],
            index=0,
        )
    
    col_disp1, col_disp2 = st.columns(2)
    with col_disp1:
        touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
    with col_disp2:
        ips = st.selectbox("IPS Display", ["No", "Yes"])

    col_stor1, col_stor2 = st.columns(2)
    with col_stor1:
        hdd = st.selectbox("HDD Storage (GB)", sorted(df["HDD"].unique()), index=0)
    with col_stor2:
        ssd = st.selectbox("SSD Storage (GB)", sorted(df["SSD"].unique()), index=sorted(df["SSD"].unique()).index(256) if 256 in df["SSD"].unique() else 0)

# Calculate PPI from resolution and screen size
res_str = resolution.split()[0]
x_res = int(res_str.split("x")[0])
y_res = int(res_str.split("x")[1])
ppi = float(((x_res ** 2) + (y_res ** 2)) ** 0.5 / screen_size)

st.markdown("<br>", unsafe_allow_html=True)

# --- Prediction Action ---
if st.button("🚀 Calculate Estimated Price", use_container_width=True, type="primary"):
    touchscreen_val = 1 if touchscreen == "Yes" else 0
    ips_val = 1 if ips == "Yes" else 0

    query = pd.DataFrame(
        [[company, type_name, ram, touchscreen_val, ips_val, ppi, cpu_brand, hdd, ssd, gpu_brand, os_type]],
        columns=[
            "Company", "TypeName", "Ram", "Touchscreen", "IPS", "ppi", "Cpu_brand", "HDD", "SSD", "brand_gpu", "os"
        ],
    )

    try:
        log_price = pipe.predict(query)[0]
        predicted_price = int(np.exp(log_price))

        st.markdown(
            f"""
            <div class="price-card">
                <div class="price-label">Estimated Market Price</div>
                <div class="price-value">₹ {predicted_price:,}</div>
                <div style="margin-top: 8px; font-size: 0.85rem; opacity: 0.8;">
                    Calculated for: {company} {type_name} • {ram}GB RAM • {cpu_brand} • {ssd}GB SSD • {res_str} ({screen_size}")
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Error calculating price: {e}")
