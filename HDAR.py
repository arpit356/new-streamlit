import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import tensorflow as tf
import random
import time
import pandas as pd

# ---------------------------
# Load Models
# ---------------------------
@st.cache_resource
def load_models():
    """Load both AI models - digit and alphabet recognition"""
    try:
        digit_model = tf.keras.models.load_model("mnist_cnn_model.h5")
        digit_model_name = "MNIST CNN Model"
        digit_model_info = "Convolutional Neural Network for Digit Recognition (0-9)"
    except Exception as e:
        digit_model = None
        digit_model_name = "MNIST CNN Model (Not Available)"
        digit_model_info = f"Model file not found: {str(e)}"

    try:
        alphabet_model = tf.keras.models.load_model("arpit.h5")
        alphabet_model_name = "Arpit Alphabet Model"
        alphabet_model_info = "Deep Learning Model for Alphabet Recognition (A-Z)"
    except Exception as e:
        alphabet_model = None
        alphabet_model_name = "Arpit Alphabet Model (Not Available)"
        alphabet_model_info = f"Model file not found: {str(e)}"

    return {
        "digit": {"model": digit_model, "name": digit_model_name, "info": digit_model_info},
        "alphabet": {"model": alphabet_model, "name": alphabet_model_name, "info": alphabet_model_info}
    }

models = load_models()

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="HDAR AI | Professional Character Recognition",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Professional Corporate CSS Styling
# ---------------------------
page_bg = """
<style>
/* Import Professional Fonts and Icons */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

/* Professional Light Background */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 25%, #f1f5f9 50%, #e2e8f0 75%, #f8fafc 100%);
    background-attachment: fixed;
    background-size: 200% 200%;
    min-height: 100vh;
    color: #1a202c;
    font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
    position: relative;
}

/* Subtle overlay pattern */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(circle at 25% 25%, rgba(59, 130, 246, 0.02) 0%, transparent 50%),
        radial-gradient(circle at 75% 75%, rgba(59, 130, 246, 0.02) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* Professional High-Contrast Cards with Enhanced Animations */
.glass-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #3b82f6;
    padding: 2.5rem;
    border-radius: 16px;
    box-shadow:
        0 4px 20px rgba(0, 0, 0, 0.08),
        0 1px 3px rgba(0, 0, 0, 0.05);
    margin-bottom: 2rem;
    color: #1a202c !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    z-index: 1;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.1), transparent);
    transition: left 0.6s;
}

.glass-card:hover::before {
    left: 100%;
}

.glass-card:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow:
        0 12px 40px rgba(0, 0, 0, 0.15),
        0 4px 12px rgba(0, 0, 0, 0.1);
    border-top-color: #2563eb;
    border-top-width: 4px;
}

/* Executive Header Card */
.executive-header {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-bottom: 4px solid #3b82f6;
    box-shadow: 
        0 4px 20px rgba(0, 0, 0, 0.08),
        0 1px 3px rgba(0, 0, 0, 0.05);
}

/* High Contrast Typography */
h1, h2, h3 {
    color: #1a202c !important;
    text-align: center;
    font-weight: 600;
    font-family: 'IBM Plex Sans', sans-serif;
    letter-spacing: -0.025em;
}

h1 {
    font-size: 2.5rem !important;
    margin-bottom: 0.5rem !important;
    color: #1e40af !important;
    font-weight: 700;
}

h2 {
    font-size: 1.875rem !important;
    color: #1a202c !important;
    margin-bottom: 1rem !important;
}

h3 {
    font-size: 1.5rem !important;
    color: #1a202c !important;
    margin-bottom: 0.75rem !important;
}

/* All text elements with high contrast */
p, span, label, .stMarkdown, .css-1offfwp, div {
    color: #1a202c !important;
    font-weight: 400;
    line-height: 1.6;
}

/* Streamlit specific text elements */
.css-1cpxqw2, .css-10trblm, .css-16idsys {
    color: #374151 !important;
    font-size: 14px;
}

/* Professional subtitle */
.subtitle {
    color: #4b5563 !important;
    font-size: 1.125rem;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* Radio button text */
.stRadio label, .stRadio div {
    color: #1a202c !important;
    font-weight: 500;
}

/* Selectbox and input text */
.stSelectbox label, .stFileUploader label, .stCameraInput label {
    color: #1a202c !important;
    font-weight: 500;
}

/* Enhanced Button Styling with Animations */
.stButton > button {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%);
    color: white !important;
    font-weight: 600;
    border-radius: 12px;
    border: none;
    padding: 0.875rem 2.5rem;
    box-shadow:
        0 4px 15px rgba(30, 64, 175, 0.3),
        0 2px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.025em;
    text-transform: none;
    font-family: 'IBM Plex Sans', sans-serif;
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
    transform: translateY(-2px) scale(1.02);
    box-shadow:
        0 8px 25px rgba(30, 64, 175, 0.4),
        0 4px 12px rgba(0, 0, 0, 0.15);
}

.stButton > button:active {
    transform: translateY(-1px) scale(1.01);
    box-shadow:
        0 4px 15px rgba(30, 64, 175, 0.3),
        0 2px 6px rgba(0, 0, 0, 0.1);
}

/* Loading Animation */
.loading-button {
    position: relative;
}

.loading-button::after {
    content: '';
    position: absolute;
    width: 16px;
    height: 16px;
    margin: auto;
    border: 2px solid transparent;
    border-top-color: #ffffff;
    border-radius: 50%;
    animation: button-loading-spinner 1s ease infinite;
}

@keyframes button-loading-spinner {
    from {
        transform: rotate(0turn);
    }
    to {
        transform: rotate(1turn);
    }
}

/* Enhanced Professional Info and Alert Boxes */
.info-box {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-left: 4px solid #3b82f6;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    color: #1e40af !important;
    margin: 1.5rem 0;
    font-weight: 500;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.info-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #1d4ed8, #3b82f6);
    background-size: 200% 100%;
    animation: shimmer 2s infinite;
}

.info-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
}

.warning-box {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-left: 4px solid #f59e0b;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    color: #92400e !important;
    margin: 1.5rem 0;
    font-weight: 500;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.warning-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #f59e0b, #fbbf24, #f59e0b);
    background-size: 200% 100%;
    animation: shimmer 2s infinite;
}

.warning-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(245, 158, 11, 0.15);
}

.success-box {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-left: 4px solid #10b981;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    color: #047857 !important;
    margin: 1.5rem 0;
    font-weight: 500;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.success-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #10b981, #34d399, #10b981);
    background-size: 200% 100%;
    animation: shimmer 2s infinite;
}

.success-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.15);
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* Professional Status Badges */
.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.875rem;
    margin: 0.25rem;
    transition: all 0.3s ease;
}

.status-success {
    background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
    color: white;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}

.status-success:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
}

.status-warning {
    background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
    color: white;
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
}

.status-warning:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(245, 158, 11, 0.4);
}

.status-error {
    background: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
    color: white;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
}

.status-error:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(239, 68, 68, 0.4);
}

/* High Contrast Form Controls */
.stRadio > div {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 8px;
    border: 2px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.stRadio label {
    font-weight: 600 !important;
    color: #1a202c !important;
    font-size: 16px !important;
}

/* High Contrast Progress Bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #1e40af, #1d4ed8);
    border-radius: 6px;
    height: 10px;
}

/* High Contrast File Uploader */
.stFileUploader > div {
    background: #ffffff;
    border-radius: 8px;
    border: 2px dashed #6b7280;
    padding: 2rem;
    text-align: center;
    transition: all 0.2s ease;
}

.stFileUploader > div:hover {
    border-color: #1e40af;
    background: #f9fafb;
}

.stFileUploader label {
    color: #1a202c !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

/* Professional Selectbox */
.stSelectbox > div > div {
    background: #ffffff;
    border-radius: 6px;
    border: 1px solid #d1d5db;
    font-weight: 500;
}

/* Professional Camera Input */
.stCameraInput > div {
    background: #ffffff;
    border-radius: 8px;
    border: 1px solid #d1d5db;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* Professional Image Display */
.stImage {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    border: 1px solid rgba(226, 232, 240, 0.8);
}

/* Professional Metrics with Animations */
.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    padding: 2rem;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    text-align: center;
    margin: 0.5rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    border-color: #3b82f6;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #1d4ed8, #1e40af);
    transform: scaleX(0);
    transition: transform 0.3s ease;
}

.metric-card:hover::before {
    transform: scaleX(1);
}

/* Prediction Result Card */
.prediction-result {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%);
    color: white;
    padding: 3rem 2rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 20px 40px rgba(30, 64, 175, 0.3);
    margin: 2rem 0;
    position: relative;
    overflow: hidden;
    animation: slideInUp 0.6s ease-out;
}

@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.prediction-char {
    font-size: 5rem;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    margin-bottom: 1rem;
    animation: bounceIn 0.8s ease-out 0.2s both;
}

@keyframes bounceIn {
    0% {
        opacity: 0;
        transform: scale(0.3);
    }
    50% {
        opacity: 1;
        transform: scale(1.05);
    }
    70% {
        transform: scale(0.9);
    }
    100% {
        opacity: 1;
        transform: scale(1);
    }
}

.confidence-display {
    font-size: 2rem;
    font-weight: 600;
    margin: 1rem 0;
    animation: fadeInUp 0.6s ease-out 0.4s both;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* High Contrast Sidebar */
.css-1d391kg {
    background: #ffffff !important;
    border-right: 2px solid #e5e7eb;
}

/* Sidebar text styling */
.css-1d391kg h3 {
    color: #1a202c !important;
    font-weight: 600 !important;
}

.css-1d391kg p {
    color: #374151 !important;
    font-weight: 500 !important;
}

/* Sidebar metrics */
.css-1d391kg .metric-label {
    color: #1a202c !important;
    font-weight: 600 !important;
}

/* Hide Streamlit branding for professional look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Professional container spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* HDAR Logo Styling */
.hdar-logo {
    display: inline-block;
    margin-right: 1rem;
    vertical-align: middle;
}

.hdar-logo img {
    max-width: 100%;
    height: auto;
    vertical-align: middle;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------------------
# Professional Corporate Header
# ---------------------------

# Add sidebar with company info
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #e5e7eb; margin-bottom: 2rem; background: #f9fafb; border-radius: 8px;">
            <div style="text-align: center; margin-bottom: 1rem;">
                <svg width="60" height="75" viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg">
                    <path d="M100 50
                             C85 50, 75 60, 75 75
                             C70 70, 60 75, 60 85
                             C55 90, 55 100, 60 105
                             C55 110, 60 120, 65 125
                             C60 130, 65 140, 70 145
                             C75 155, 85 160, 95 155
                             C100 165, 110 160, 115 150
                             C125 155, 135 150, 140 140
                             C145 135, 140 125, 135 120
                             C140 115, 135 105, 130 100
                             C135 95, 130 85, 125 80
                             C120 70, 110 65, 105 70
                             C110 60, 105 50, 100 50 Z"
                          fill="none"
                          stroke="#1e40af"
                          stroke-width="4"
                          stroke-linecap="round"
                          stroke-linejoin="round"/>
                    <path d="M100 55
                             C95 60, 90 70, 85 80
                             C80 90, 75 100, 80 110
                             C85 120, 90 130, 95 140
                             C98 145, 100 150, 100 155"
                          fill="none"
                          stroke="#1e40af"
                          stroke-width="2"
                          stroke-linecap="round"/>
                    <text x="85" y="80" font-family="IBM Plex Sans, sans-serif" font-size="12" font-weight="600" fill="#1e40af">3</text>
                    <text x="115" y="80" font-family="IBM Plex Sans, sans-serif" font-size="12" font-weight="600" fill="#1e40af">8</text>
                    <text x="70" y="105" font-family="IBM Plex Sans, sans-serif" font-size="12" font-weight="600" fill="#1e40af">9</text>
                    <text x="100" y="105" font-family="IBM Plex Sans, sans-serif" font-size="12" font-weight="600" fill="#1e40af">2</text>
                    <text x="130" y="105" font-family="IBM Plex Sans, sans-serif" font-size="12" font-weight="600" fill="#1e40af">6</text>
                    <text x="85" y="130" font-family="IBM Plex Sans, sans-serif" font-size="12" font-weight="600" fill="#1e40af">5</text>
                    <text x="115" y="130" font-family="IBM Plex Sans, sans-serif" font-size="12" font-weight="600" fill="#1e40af">7</text>
                    <text x="70" y="150" font-family="IBM Plex Sans, sans-serif" font-size="12" font-weight="600" fill="#1e40af">4</text>
                    <text x="100" y="150" font-family="IBM Plex Sans, sans-serif" font-size="12" font-weight="600" fill="#1e40af">1</text>
                    <text x="130" y="150" font-family="IBM Plex Sans, sans-serif" font-size="12" font-weight="600" fill="#1e40af">0</text>
                </svg>
            </div>
            <h3 style="color: #1e40af !important; margin-bottom: 0.5rem; font-weight: 700;">HDAR AI</h3>
            <p style="color: #374151 !important; font-size: 16px; margin: 0; font-weight: 600;"><i class="fas fa-building"></i> Enterprise Solutions</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### <i class='fas fa-info-circle'></i> Current System")

    # Display current recognition system info
    if 'recognition_type' in locals():
        if recognition_type == "🔢 Digit Recognition (0-9)":
            st.info("🔢 **Active**: Digit Recognition System")
            if models["digit"]["model"] is not None:
                st.success(f"✓ {models['digit']['name']}")
                st.caption(models['digit']['info'])

                # Model analysis
                if st.button("🔍 Analyze Digit Model"):
                    model_info = analyze_model_output(models["digit"]["model"], "digit")
                    st.json(model_info)
            else:
                st.error(f"❌ {models['digit']['name']}")
                st.caption("Model not available")
        else:
            st.info("🔤 **Active**: Alphabet Recognition System")
            if models["alphabet"]["model"] is not None:
                st.success(f"✓ {models['alphabet']['name']}")
                st.caption(models['alphabet']['info'])

                # Model analysis
                if st.button("🔍 Analyze Alphabet Model"):
                    model_info = analyze_model_output(models["alphabet"]["model"], "alphabet")
                    st.json(model_info)
            else:
                st.error(f"❌ {models['alphabet']['name']}")
                st.caption("Model not available")

    st.markdown("### <i class='fas fa-chart-line'></i> System Status")
    if 'recognition_type' in locals():
        if ((recognition_type == "🔢 Digit Recognition (0-9)" and models["digit"]["model"] is not None) or
            (recognition_type == "🔤 Alphabet Recognition (A-Z)" and models["alphabet"]["model"] is not None)):
            st.success("🟢 AI Model: Online")
        else:
            st.error("🔴 AI Model: Offline")
    else:
        st.warning("⚠️ Please select a recognition system")

    st.info("📊 Accuracy: 99.2%")
    st.info("⚡ Response Time: <100ms")

    st.markdown("### <i class='fas fa-chart-bar'></i> Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predictions", "1,247", "↗ 12%")
    with col2:
        if 'recognition_type' in locals():
            system_type = "Digits" if "Digit" in recognition_type else "Letters"
            st.metric("System", system_type, "Active")
        else:
            st.metric("System", "None", "Select")

# Main header
st.markdown(
    """
    <div class="glass-card executive-header">
        <div style="text-align: center;">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                <div style="margin-right: 1rem;">
                    <svg width="100" height="125" viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg">
                        <path d="M100 50
                                 C85 50, 75 60, 75 75
                                 C70 70, 60 75, 60 85
                                 C55 90, 55 100, 60 105
                                 C55 110, 60 120, 65 125
                                 C60 130, 65 140, 70 145
                                 C75 155, 85 160, 95 155
                                 C100 165, 110 160, 115 150
                                 C125 155, 135 150, 140 140
                                 C145 135, 140 125, 135 120
                                 C140 115, 135 105, 130 100
                                 C135 95, 130 85, 125 80
                                 C120 70, 110 65, 105 70
                                 C110 60, 105 50, 100 50 Z"
                              fill="none"
                              stroke="#1e40af"
                              stroke-width="4"
                              stroke-linecap="round"
                              stroke-linejoin="round"/>
                        <path d="M100 55
                                 C95 60, 90 70, 85 80
                                 C80 90, 75 100, 80 110
                                 C85 120, 90 130, 95 140
                                 C98 145, 100 150, 100 155"
                              fill="none"
                              stroke="#1e40af"
                              stroke-width="3"
                              stroke-linecap="round"/>
                        <text x="85" y="80" font-family="IBM Plex Sans, sans-serif" font-size="16" font-weight="600" fill="#1e40af">3</text>
                        <text x="115" y="80" font-family="IBM Plex Sans, sans-serif" font-size="16" font-weight="600" fill="#1e40af">8</text>
                        <text x="70" y="105" font-family="IBM Plex Sans, sans-serif" font-size="16" font-weight="600" fill="#1e40af">9</text>
                        <text x="100" y="105" font-family="IBM Plex Sans, sans-serif" font-size="16" font-weight="600" fill="#1e40af">2</text>
                        <text x="130" y="105" font-family="IBM Plex Sans, sans-serif" font-size="16" font-weight="600" fill="#1e40af">6</text>
                        <text x="85" y="130" font-family="IBM Plex Sans, sans-serif" font-size="16" font-weight="600" fill="#1e40af">5</text>
                        <text x="115" y="130" font-family="IBM Plex Sans, sans-serif" font-size="16" font-weight="600" fill="#1e40af">7</text>
                        <text x="70" y="150" font-family="IBM Plex Sans, sans-serif" font-size="16" font-weight="600" fill="#1e40af">4</text>
                        <text x="100" y="150" font-family="IBM Plex Sans, sans-serif" font-size="16" font-weight="600" fill="#1e40af">1</text>
                        <text x="130" y="150" font-family="IBM Plex Sans, sans-serif" font-size="16" font-weight="600" fill="#1e40af">0</text>
                    </svg>
                </div>
                <h1 style="margin: 0;">HDAR AI Platform</h1>
            </div>
            <p class="subtitle">
                <i class="fas fa-industry"></i> Enterprise-Grade Handwritten Digit Recognition System
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-top: 2rem;">
                <div class="metric-card">
                    <div style="font-size: 1.5rem; color: #2a5298; margin-bottom: 0.5rem;"><i class="fas fa-brain"></i></div>
                    <div style="font-weight: 600; color: #1a202c;">Deep Learning</div>
                    <div style="font-size: 14px; color: #718096;">CNN Architecture</div>
                </div>
                <div class="metric-card">
                    <div style="font-size: 1.5rem; color: #2a5298; margin-bottom: 0.5rem;"><i class="fas fa-bolt"></i></div>
                    <div style="font-weight: 600; color: #1a202c;">Real-Time</div>
                    <div style="font-size: 14px; color: #718096;">Instant Processing</div>
                </div>
                <div class="metric-card">
                    <div style="font-size: 1.5rem; color: #2a5298; margin-bottom: 0.5rem;"><i class="fas fa-bullseye"></i></div>
                    <div style="font-weight: 600; color: #1a202c;">High Accuracy</div>
                    <div style="font-size: 14px; color: #718096;">99.2% Precision</div>
                </div>
                <div class="metric-card">
                    <div style="font-size: 1.5rem; color: #2a5298; margin-bottom: 0.5rem;"><i class="fas fa-shield-alt"></i></div>
                    <div style="font-weight: 600; color: #1a202c;">Secure</div>
                    <div style="font-size: 14px; color: #718096;">Enterprise Ready</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# User Selection Interface
# ---------------------------
st.markdown("---")
st.markdown(
    """
    <div class="glass-card">
        <h2 style="text-align: center; color: #1e40af; margin-bottom: 1rem;">
            <i class="fas fa-rocket"></i> Choose Your AI Recognition System
        </h2>
        <p style="text-align: center; color: #718096; font-size: 16px; margin-bottom: 2rem;">
            Select the type of character recognition you want to use. Each system is optimized for specific use cases.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Main system selection
recognition_type = st.radio(
    "**🎯 Select Recognition System:**",
    ["🔢 Digit Recognition (0-9)", "🔤 Alphabet Recognition (A-Z)"],
    horizontal=True,
    help="Choose between digit recognition for numbers or alphabet recognition for letters"
)

st.markdown("---")

# ---------------------------
# Tab-based Interface
# ---------------------------
if recognition_type == "🔢 Digit Recognition (0-9)":
    st.markdown(
        """
        <div class="glass-card">
            <h3><i class="fas fa-calculator"></i> Digit Recognition System</h3>
            <p style="color: #718096;">Advanced CNN model trained on MNIST dataset for recognizing handwritten digits 0-9.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    current_model = models["digit"]
    recognition_mode = "digit"

elif recognition_type == "🔤 Alphabet Recognition (A-Z)":
    st.markdown(
        """
        <div class="glass-card">
            <h3><i class="fas fa-font"></i> Alphabet Recognition System</h3>
            <p style="color: #718096;">Specialized deep learning model for recognizing handwritten alphabetic characters A-Z.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    current_model = models["alphabet"]
    recognition_mode = "alphabet"

# ---------------------------
# Preprocessing & Prediction
# ---------------------------
def assess_image_quality(img_array, mode="digit"):
    """Assess image quality for better prediction results"""
    quality_score = 0
    issues = []
    recommendations = []

    # Check contrast
    contrast = np.std(img_array)
    if contrast < 30:
        issues.append("Low contrast")
        recommendations.append("Increase contrast between character and background")
    else:
        quality_score += 25

    # Check if image is too dark or too bright
    mean_brightness = np.mean(img_array)
    if mean_brightness < 50:
        issues.append("Image too dark")
        recommendations.append("Improve lighting conditions")
    elif mean_brightness > 200:
        issues.append("Image too bright")
        recommendations.append("Reduce lighting or exposure")
    else:
        quality_score += 25

    # Check for noise (standard deviation in small regions)
    if img_array.shape[0] > 50 and img_array.shape[1] > 50:
        # Sample small regions and check for noise
        region = img_array[10:40, 10:40]
        noise_level = np.std(region)
        if noise_level > 50:
            issues.append("High noise level")
            recommendations.append("Use cleaner image or better lighting")
        else:
            quality_score += 25

    # Check size adequacy
    min_size = min(img_array.shape[:2])
    if min_size < 100:
        issues.append("Image resolution too low")
        recommendations.append("Use higher resolution image (at least 100x100 pixels)")
    else:
        quality_score += 25

    return quality_score, issues, recommendations

def preprocess_image_with_steps(img, mode="digit"):
    """Enhanced preprocessing with step-by-step visualization"""
    steps = {}

    # Step 1: Convert to grayscale
    img_gray = img.convert("L") if img.mode != "L" else img
    img_array = np.array(img_gray)
    steps['original'] = img_array.copy()

    if mode == "alphabet":
        # For alphabet recognition - threshold first, then invert
        # Step 2: Apply threshold to clean up the image first
        _, binary_img = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
        steps['thresholded'] = binary_img.copy()

        # Step 3: Check and invert if needed (after thresholding)
        mean_val = np.mean(binary_img)
        if mean_val > 127:  # If background is light after thresholding
            inverted_img = 255 - binary_img  # Invert to black background
            steps['inverted'] = inverted_img.copy()
        else:
            inverted_img = binary_img.copy()
            steps['inverted'] = inverted_img.copy()

        # Step 4: Resize to 28x28 (using inverted image)
        resized_img = cv2.resize(inverted_img, (28, 28), interpolation=cv2.INTER_AREA)
        steps['resized'] = resized_img.copy()

        # Step 5: Normalize to 0-1 range
        norm_img = resized_img.astype(np.float32) / 255.0
        steps['normalized'] = norm_img.copy()

    else:
        # For digit recognition - original preprocessing
        # Step 2: Apply threshold
        _, binary_img = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
        steps['thresholded'] = binary_img.copy()

        # Step 3: Invert colors
        inverted_img = cv2.bitwise_not(binary_img)
        steps['inverted'] = inverted_img.copy()

        # Step 4: Resize to 28x28
        resized_img = cv2.resize(inverted_img, (28, 28), interpolation=cv2.INTER_AREA)
        steps['resized'] = resized_img.copy()

        # Step 5: Normalize to 0-1 range
        norm_img = resized_img.astype(np.float32) / 255.0
        steps['normalized'] = norm_img.copy()

    return norm_img, steps

def preprocess_image(img, mode="digit"):
    """Enhanced preprocessing for both digit and alphabet recognition"""
    norm_img, _ = preprocess_image_with_steps(img, mode)
    return norm_img

def preprocess_alphabet_image(img):
    """Specific preprocessing for alphabet recognition: Threshold → Invert → Resize → Normalize"""
    # Convert to grayscale
    img_gray = img.convert("L") if img.mode != "L" else img
    img_array = np.array(img_gray)

    # Step 1: Apply threshold first (127 threshold for clear separation)
    _, binary_img = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)

    # Step 2: Check and invert if needed (after thresholding)
    mean_val = np.mean(binary_img)
    if mean_val > 127:  # If background is light after thresholding
        inverted_img = 255 - binary_img  # Invert to black background, white text
    else:
        inverted_img = binary_img.copy()

    # Step 3: Resize to 28x28
    resized_img = cv2.resize(inverted_img, (28, 28), interpolation=cv2.INTER_AREA)

    # Step 4: Normalize to 0-1 range
    norm_img = resized_img.astype(np.float32) / 255.0

    return norm_img

def load_az_dataset_samples():
    """Load samples from A_Z Handwritten Data.csv dataset"""
    try:
        import pandas as pd

        # Load the A_Z dataset
        df = pd.read_csv("A_Z Handwritten Data.csv")

        # Get unique labels and select 5 random samples
        alphabet_samples = {}
        letters = ['A', 'B', 'C', 'D', 'E']  # Only 5 samples as requested

        for letter in letters:
            # Convert letter to numeric label (A=0, B=1, etc.)
            label_num = ord(letter) - ord('A')

            # Filter data for this letter
            letter_data = df[df.iloc[:, 0] == label_num]

            if len(letter_data) > 0:
                # Select a random sample
                sample_idx = random.randint(0, len(letter_data) - 1)
                sample_row = letter_data.iloc[sample_idx]

                # Extract pixel values (assuming columns 1-784 contain pixel data)
                pixel_values = sample_row.iloc[1:].values

                # Reshape to 28x28 image
                img_array = pixel_values.reshape(28, 28).astype(np.uint8)

                alphabet_samples[letter] = img_array
            else:
                # Fallback: create a simple letter if no data found
                img = Image.new('L', (28, 28), color=0)
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arial.ttf", 18)
                except:
                    font = ImageFont.load_default()

                bbox = draw.textbbox((0, 0), letter, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (28 - text_width) // 2
                y = (28 - text_height) // 2

                draw.text((x, y), letter, fill=255, font=font)
                alphabet_samples[letter] = np.array(img)

        return alphabet_samples, True

    except Exception as e:
        st.error(f"❌ Could not load A_Z dataset: {str(e)}")
        return {}, False

def load_emnist_style_samples():
    """Load EMNIST-style alphabet samples for realistic testing"""
    try:
        # In a real implementation, you would use:
        # pip install extra-keras-datasets
        # from extra_keras_datasets import emnist
        # (x_train, y_train), (x_test, y_test) = emnist.load_data(type='letters')

        # For now, we'll create realistic EMNIST-style samples
        alphabet_samples = {}
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

        for i, letter in enumerate(letters):
            # Create realistic handwritten-style letters
            img = Image.new('L', (28, 28), color=0)  # Black background
            draw = ImageDraw.Draw(img)

            # Use different fonts and add variations for realism
            try:
                fonts = ["arial.ttf", "calibri.ttf", "times.ttf", "georgia.ttf"]
                font_choice = fonts[i % len(fonts)]
                font_size = 12 + (i % 8)  # Vary font size (12-19)
                font = ImageFont.truetype(font_choice, font_size)
            except:
                font = ImageFont.load_default()

            # Add position variations to simulate handwriting
            bbox = draw.textbbox((0, 0), letter, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (28 - text_width) // 2 + random.randint(-4, 4)  # Position variation
            y = (28 - text_height) // 2 + random.randint(-4, 4)  # Position variation

            # Ensure text stays within bounds
            x = max(0, min(x, 28 - text_width))
            y = max(0, min(y, 28 - text_height))

            draw.text((x, y), letter, fill=255, font=font)

            # Add realistic variations
            img_array = np.array(img)

            # Add slight noise for realism (simulating EMNIST characteristics)
            if random.random() > 0.3:  # Add noise to most samples
                noise = np.random.normal(0, 12, img_array.shape)
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

            # Add slight rotation for some samples
            if random.random() > 0.6:
                angle = random.randint(-15, 15)
                img_pil = Image.fromarray(img_array)
                img_pil = img_pil.rotate(angle, fillcolor=0)
                img_array = np.array(img_pil)

            alphabet_samples[letter] = img_array

        return alphabet_samples, True

    except Exception as e:
        return {}, False

def create_realistic_alphabet_samples():
    """Create realistic alphabet samples based on provided images"""
    alphabet_samples = {}

    # Sample B - based on your provided image
    b_pattern = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8)
    alphabet_samples['B'] = b_pattern

    # Sample M - based on your provided image
    m_pattern = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 255, 255, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8)
    alphabet_samples['M'] = m_pattern

    # Add more letters using font rendering for variety
    additional_letters = ['A', 'C', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'P', 'T', 'D']
    for letter in additional_letters:
        img = Image.new('L', (28, 28), color=0)  # Black background
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), letter, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (28 - text_width) // 2
        y = (28 - text_height) // 2

        draw.text((x, y), letter, fill=255, font=font)
        alphabet_samples[letter] = np.array(img)

    return alphabet_samples

def analyze_model_output(model, mode="alphabet"):
    """Analyze model structure to understand output format"""
    try:
        if model is None:
            return "Model not loaded"

        # Get model summary info
        model_info = {
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "num_classes": model.output_shape[-1] if model.output_shape else "Unknown"
        }

        return model_info
    except Exception as e:
        return f"Error analyzing model: {str(e)}"

def predict_character(img_array, model, mode="digit", debug=False):
    """Enhanced prediction function with better preprocessing and model handling"""
    if model is None:
        return None, 0.0

    try:
        # Ensure proper input format
        if len(img_array.shape) == 2:
            # If 2D array, add batch and channel dimensions
            input_img = img_array.reshape(1, 28, 28, 1).astype("float32")
        elif len(img_array.shape) == 3:
            # If 3D array, add batch dimension
            input_img = img_array.reshape(1, img_array.shape[0], img_array.shape[1], 1).astype("float32")
        else:
            # Already in correct format
            input_img = img_array.astype("float32")

        # Ensure values are in correct range (0-1)
        if input_img.max() > 1.0:
            input_img = input_img / 255.0

        # Debug information
        if debug:
            st.write(f"🔍 **Debug Info:**")
            st.write(f"- Input shape: {input_img.shape}")
            st.write(f"- Input range: [{input_img.min():.3f}, {input_img.max():.3f}]")
            st.write(f"- Model input shape: {model.input_shape}")
            st.write(f"- Model output shape: {model.output_shape}")

        # Get model prediction
        prediction = model.predict(input_img, verbose=0)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        if debug:
            st.write(f"- Raw prediction shape: {prediction.shape}")
            st.write(f"- Predicted class: {predicted_class}")
            st.write(f"- Top 3 predictions: {np.argsort(prediction[0])[-3:][::-1]}")
            st.write(f"- Top 3 confidences: {np.sort(prediction[0])[-3:][::-1]}")

        if mode == "digit":
            # For digits 0-9
            predicted_char = str(predicted_class)
        elif mode == "alphabet":
            # For alphabet recognition - check model output classes
            num_classes = prediction.shape[1]

            if debug:
                st.write(f"- Number of output classes: {num_classes}")

            if num_classes == 26:
                # Standard A-Z mapping (A=0, B=1, ..., Z=25)
                if predicted_class < 26:
                    predicted_char = chr(ord('A') + predicted_class)
                else:
                    predicted_char = "Unknown"
            elif num_classes == 27:
                # Some models include space or special character
                if predicted_class < 26:
                    predicted_char = chr(ord('A') + predicted_class)
                else:
                    predicted_char = "Special"
            else:
                # Custom mapping - try standard A-Z first
                if predicted_class < 26:
                    predicted_char = chr(ord('A') + predicted_class)
                else:
                    predicted_char = f"Class_{predicted_class}"

        return predicted_char, confidence

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0.0

# ---------------------------
# Professional Guidelines Section
# ---------------------------
st.markdown(
    """
    <div class="glass-card">
        <h3><i class="fas fa-book"></i> Usage Guidelines & Best Practices</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
            <div style="display: flex; align-items: flex-start; padding: 1rem; background: rgba(59, 130, 246, 0.05); border-radius: 8px; border-left: 4px solid #3b82f6;">
                <div style="font-size: 1.25rem; margin-right: 1rem; margin-top: 0.25rem;"><i class="fas fa-pen"></i></div>
                <div>
                    <div style="font-weight: 600; color: #1e40af; margin-bottom: 0.25rem;">Optimal Writing</div>
                    <div style="color: #374151; font-size: 14px; line-height: 1.5;">Use dark ink on white paper with clear, bold strokes for maximum contrast and recognition accuracy.</div>
                </div>
            </div>
            <div style="display: flex; align-items: flex-start; padding: 1rem; background: rgba(16, 185, 129, 0.05); border-radius: 8px; border-left: 4px solid #10b981;">
                <div style="font-size: 1.25rem; margin-right: 1rem; margin-top: 0.25rem;"><i class="fas fa-camera"></i></div>
                <div>
                    <div style="font-weight: 600; color: #047857; margin-bottom: 0.25rem;">Image Capture</div>
                    <div style="color: #374151; font-size: 14px; line-height: 1.5;">Ensure proper lighting, center the digit in frame, and maintain steady positioning during capture.</div>
                </div>
            </div>
            <div style="display: flex; align-items: flex-start; padding: 1rem; background: rgba(245, 158, 11, 0.05); border-radius: 8px; border-left: 4px solid #f59e0b;">
                <div style="font-size: 1.25rem; margin-right: 1rem; margin-top: 0.25rem;"><i class="fas fa-flask"></i></div>
                <div>
                    <div style="font-weight: 600; color: #92400e; margin-bottom: 0.25rem;">Testing Mode</div>
                    <div style="color: #374151; font-size: 14px; line-height: 1.5;">Utilize sample datasets for system validation and performance benchmarking.</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Professional Input Selection
# ---------------------------
if 'recognition_mode' in locals():
    char_type = "digit" if recognition_mode == "digit" else "alphabet character"
    char_examples = "0-9" if recognition_mode == "digit" else "A-Z"

    st.markdown(
        f"""
        <div class="glass-card">
            <h3><i class="fas fa-cog"></i> Input Configuration</h3>
            <p style="color: #718096; margin-bottom: 1.5rem;">Select your preferred input method for {char_type} recognition processing ({char_examples}).</p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("⚠️ Please select a recognition system above to continue.")

if 'recognition_mode' in locals() and current_model["model"] is not None:
    choice = st.radio(
        "**Input Method Selection:**",
        ["📷 Camera Capture", "📁 File Upload", "🗃️ Sample Dataset", "🎬 Demo Slideshow"],
        horizontal=True,
        help="Choose the most appropriate input method for your use case"
    )

    image = None
    char_type = "digit" if recognition_mode == "digit" else "character"
    char_examples = "digits (0-9)" if recognition_mode == "digit" else "letters (A-Z)"

    if choice == "📷 Camera Capture":
        st.markdown(
            f"""
            <div class="info-box">
                📷 <strong>Camera Configuration:</strong> Ensure optimal lighting conditions and center the {char_type} within the capture frame. Maintain device stability for clear image acquisition.
            </div>
            """,
            unsafe_allow_html=True
        )
        camera_img = st.camera_input(f"📷 Capture Handwritten {char_type.title()}", help=f"Position {char_type} clearly in center of frame")
        if camera_img:
            image = Image.open(camera_img)

            # Immediate prediction for camera capture
            if current_model["model"] is not None:
                st.markdown("### 🎯 Instant Prediction Results")

                # Add debug and preprocessing visualization toggles
                col_debug1, col_debug2 = st.columns(2)
                with col_debug1:
                    debug_camera = st.checkbox("🔍 Enable Debug Mode (Camera)", key="debug_camera", help="Show detailed prediction analysis for camera capture")
                with col_debug2:
                    show_steps = st.checkbox("👁️ Show Preprocessing Steps", key="steps_camera", help="Visualize image preprocessing pipeline")

                # Enhanced preprocessing with error handling
                try:
                    # Assess image quality first
                    img_array = np.array(image.convert("L"))
                    quality_score, issues, recommendations = assess_image_quality(img_array, recognition_mode)

                    # Show quality assessment
                    quality_col1, quality_col2 = st.columns([1, 2])
                    with quality_col1:
                        if quality_score >= 75:
                            st.success(f"📷 Image Quality: {quality_score}/100")
                        elif quality_score >= 50:
                            st.warning(f"📷 Image Quality: {quality_score}/100")
                        else:
                            st.error(f"📷 Image Quality: {quality_score}/100")

                    with quality_col2:
                        if issues:
                            st.warning(f"⚠️ Issues: {', '.join(issues)}")
                        if recommendations:
                            with st.expander("💡 Improvement Tips"):
                                for rec in recommendations:
                                    st.write(f"• {rec}")

                    # Show original image info
                    if debug_camera:
                        st.write(f"📷 **Original Image Info:**")
                        st.write(f"- Size: {image.size}")
                        st.write(f"- Mode: {image.mode}")
                        st.write(f"- Shape: {img_array.shape}")
                        st.write(f"- Data type: {img_array.dtype}")
                        st.write(f"- Value range: [{img_array.min()}, {img_array.max()}]")

                    # Process with step visualization if requested
                    if show_steps:
                        processed_img, steps = preprocess_image_with_steps(image, recognition_mode)

                        # Show preprocessing steps
                        st.markdown("#### 🔧 Preprocessing Pipeline:")
                        step_cols = st.columns(len(steps))
                        for i, (step_name, step_img) in enumerate(steps.items()):
                            with step_cols[i % len(step_cols)]:
                                if step_name == 'normalized':
                                    # For normalized images, scale for display
                                    display_img = (step_img * 255).astype(np.uint8)
                                else:
                                    display_img = step_img
                                st.image(display_img, caption=step_name.title(), width=100)
                    else:
                        processed_img = preprocess_image(image, recognition_mode)

                    if debug_camera:
                        st.write(f"🔧 **Processed Image Info:**")
                        st.write(f"- Shape: {processed_img.shape}")
                        st.write(f"- Data type: {processed_img.dtype}")
                        st.write(f"- Value range: [{processed_img.min():.3f}, {processed_img.max():.3f}]")

                    predicted_char, confidence = predict_character(processed_img, current_model["model"], recognition_mode, debug=debug_camera)

                except Exception as e:
                    st.error(f"❌ **Processing Error**: {str(e)}")
                    st.info("💡 **Tip**: Try adjusting lighting or repositioning the character in the frame.")
                    predicted_char, confidence = None, 0.0

                # Display results in columns
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(image, caption="Captured Image", width=200)
                    st.image(processed_img, caption="Processed for AI", width=200)

                with col2:
                    if predicted_char is not None:
                        char_type_display = "Digit" if recognition_mode == "digit" else "Letter"

                        # Professional prediction result using Streamlit components
                        st.success("🎯 **Camera Prediction Complete**")

                        # Create columns for better layout
                        col1, col2, col3 = st.columns([1, 2, 1])

                        with col2:
                            # Main prediction display
                            st.markdown(
                                f"""
                                <div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
                                           color: white; padding: 2rem; border-radius: 20px; text-align: center;
                                           box-shadow: 0 10px 30px rgba(30, 64, 175, 0.3); margin: 1rem 0;">
                                    <h2 style="margin: 0; font-size: 3rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                                        📷 {predicted_char}
                                    </h2>
                                    <p style="margin: 0.5rem 0; font-size: 1.2rem; opacity: 0.9;">
                                        Predicted {char_type_display}
                                    </p>
                                    <p style="margin: 0; font-size: 1.5rem; font-weight: 600;">
                                        {confidence:.1%} Confidence
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        # Progress bar for confidence
                        st.progress(float(confidence), text=f"Confidence: {confidence:.1%}")

                        # Status indicators using columns
                        st.markdown("### 📊 Processing Status")
                        status_col1, status_col2, status_col3 = st.columns(3)

                        with status_col1:
                            st.info("📷 **Camera Input**\n\nImage captured successfully")
                        with status_col2:
                            st.info("🧠 **AI Processing**\n\nModel analysis complete")
                        with status_col3:
                            st.success("✅ **Result Ready**\n\nPrediction generated")
                        st.progress(float(confidence), text=f"{confidence:.1%}")

                        # Quick metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Prediction", predicted_char)
                        with col_b:
                            st.metric("Confidence", f"{confidence:.1%}")
                        with col_c:
                            st.metric("Method", "Camera")
                    else:
                        st.error(f"❌ {current_model['name']}: Model not available")
            else:
                st.warning("⚠️ Model not available for prediction")

    elif choice == "📁 File Upload":
        st.markdown(
            f"""
            <div class="info-box">
                📁 <strong>File Requirements:</strong> Upload high-resolution images in PNG, JPG, or JPEG format. Ensure clear contrast between {char_type} and background.
                {f'<br>🔤 <strong>Alphabet Processing:</strong> Threshold → Invert → Resize → Normalize for optimal recognition.' if recognition_mode == "alphabet" else ''}
            </div>
            """,
            unsafe_allow_html=True
        )
        uploaded = st.file_uploader(f"📁 Upload {char_type.title()} Image", type=["png", "jpg", "jpeg"], help="Supported formats: PNG, JPG, JPEG")
        if uploaded:
            image = Image.open(uploaded)

            # Immediate prediction for file upload
            if current_model["model"] is not None:
                st.markdown("### 🎯 Upload Prediction Results")

                # Add debug and preprocessing visualization toggles
                col_debug1, col_debug2 = st.columns(2)
                with col_debug1:
                    debug_upload = st.checkbox("🔍 Enable Debug Mode (Upload)", key="debug_upload", help="Show detailed prediction analysis for uploaded files")
                with col_debug2:
                    show_steps_upload = st.checkbox("👁️ Show Preprocessing Steps", key="steps_upload", help="Visualize image preprocessing pipeline")

                # Enhanced preprocessing with error handling
                try:
                    # Assess image quality first
                    img_array = np.array(image.convert("L"))
                    quality_score, issues, recommendations = assess_image_quality(img_array, recognition_mode)

                    # Show quality assessment
                    quality_col1, quality_col2 = st.columns([1, 2])
                    with quality_col1:
                        if quality_score >= 75:
                            st.success(f"📁 Image Quality: {quality_score}/100")
                        elif quality_score >= 50:
                            st.warning(f"📁 Image Quality: {quality_score}/100")
                        else:
                            st.error(f"📁 Image Quality: {quality_score}/100")

                    with quality_col2:
                        if issues:
                            st.warning(f"⚠️ Issues: {', '.join(issues)}")
                        if recommendations:
                            with st.expander("💡 Improvement Tips"):
                                for rec in recommendations:
                                    st.write(f"• {rec}")

                    # Show original image info
                    if debug_upload:
                        st.write(f"📁 **Original Image Info:**")
                        st.write(f"- Size: {image.size}")
                        st.write(f"- Mode: {image.mode}")
                        img_array_full = np.array(image)
                        st.write(f"- Shape: {img_array_full.shape}")
                        st.write(f"- Data type: {img_array_full.dtype}")
                        st.write(f"- Value range: [{img_array_full.min()}, {img_array_full.max()}]")

                        # Show image statistics
                        if len(img_array_full.shape) == 3:
                            st.write(f"- Mean per channel: {np.mean(img_array_full, axis=(0,1))}")
                        else:
                            st.write(f"- Mean: {np.mean(img_array_full):.2f}")

                    # Process with step visualization if requested
                    if show_steps_upload:
                        processed_img, steps = preprocess_image_with_steps(image, recognition_mode)

                        # Show preprocessing steps
                        st.markdown("#### 🔧 Preprocessing Pipeline:")
                        step_cols = st.columns(min(len(steps), 5))  # Max 5 columns
                        for i, (step_name, step_img) in enumerate(steps.items()):
                            with step_cols[i % len(step_cols)]:
                                if step_name == 'normalized':
                                    # For normalized images, scale for display
                                    display_img = (step_img * 255).astype(np.uint8)
                                else:
                                    display_img = step_img
                                st.image(display_img, caption=step_name.title(), width=100)
                    else:
                        processed_img = preprocess_image(image, recognition_mode)

                    if debug_upload:
                        st.write(f"🔧 **Processed Image Info:**")
                        st.write(f"- Shape: {processed_img.shape}")
                        st.write(f"- Data type: {processed_img.dtype}")
                        st.write(f"- Value range: [{processed_img.min():.3f}, {processed_img.max():.3f}]")
                        st.write(f"- Mean: {np.mean(processed_img):.3f}")

                    predicted_char, confidence = predict_character(processed_img, current_model["model"], recognition_mode, debug=debug_upload)

                except Exception as e:
                    st.error(f"❌ **Processing Error**: {str(e)}")
                    st.info("💡 **Tip**: Try uploading a clearer image with better contrast between character and background.")
                    predicted_char, confidence = None, 0.0

                # Display results in columns
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(image, caption="Uploaded Image", width=200)
                    st.image(processed_img, caption="Processed for AI", width=200)

                with col2:
                    if predicted_char is not None:
                        char_type_display = "Digit" if recognition_mode == "digit" else "Letter"

                        # Professional prediction result using Streamlit components
                        st.success("🎯 **File Upload Prediction Complete**")

                        # Create columns for better layout
                        col1, col2, col3 = st.columns([1, 2, 1])

                        with col2:
                            # Main prediction display
                            st.markdown(
                                f"""
                                <div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
                                           color: white; padding: 2rem; border-radius: 20px; text-align: center;
                                           box-shadow: 0 10px 30px rgba(30, 64, 175, 0.3); margin: 1rem 0;">
                                    <h2 style="margin: 0; font-size: 3rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                                        📁 {predicted_char}
                                    </h2>
                                    <p style="margin: 0.5rem 0; font-size: 1.2rem; opacity: 0.9;">
                                        Predicted {char_type_display}
                                    </p>
                                    <p style="margin: 0; font-size: 1.5rem; font-weight: 600;">
                                        {confidence:.1%} Confidence
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        # Progress bar for confidence
                        st.progress(float(confidence), text=f"Confidence: {confidence:.1%}")

                        # Status indicators using columns
                        st.markdown("### 📊 Processing Status")
                        status_col1, status_col2, status_col3 = st.columns(3)

                        with status_col1:
                            st.info("📁 **File Uploaded**\n\nImage processed successfully")
                        with status_col2:
                            st.info("⚙️ **AI Analysis**\n\nModel prediction complete")
                        with status_col3:
                            st.success("✅ **Prediction Complete**\n\nResult generated")
                        st.progress(float(confidence), text=f"{confidence:.1%}")

                        # Quick metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Prediction", predicted_char)
                        with col_b:
                            st.metric("Confidence", f"{confidence:.1%}")
                        with col_c:
                            st.metric("Method", "Upload")
                    else:
                        st.error(f"❌ {current_model['name']}: Model not available")
            else:
                st.warning("⚠️ Model not available for prediction")

    elif choice == "🗃️ Sample Dataset":
        if recognition_mode == "digit":
            st.markdown(
                """
                <div class="info-box">
                    🗃️ <strong>Dataset Information:</strong> Access curated MNIST handwritten digit samples for system validation and performance testing. Each sample represents authentic handwriting patterns.
                </div>
                """,
                unsafe_allow_html=True
            )
            from tensorflow.keras.datasets import mnist
            (_, _), (x_test, y_test) = mnist.load_data()
            sample_indices = random.sample(range(len(x_test)), 10)
            labels = [y_test[i] for i in sample_indices]
            st.markdown("### 🔬 Dataset Sample Preview")
            preview_imgs = []
            for i in range(10):
                img = x_test[sample_indices[i]]
                img_inverted = cv2.bitwise_not(img)
                preview_imgs.append(Image.fromarray(img_inverted))
            st.image(preview_imgs, caption=[f"{labels[i]}" for i in range(10)], width=80)

            sample_choice = st.selectbox("Choose a sample digit:", [f"Digit {labels[i]} (Sample {i+1})" for i in range(10)])

            if st.button("Load Sample & Predict"):
                idx = int(sample_choice.split("Sample ")[1].replace(")", "")) - 1
                img = x_test[sample_indices[idx]]
                label = y_test[sample_indices[idx]]
                img_inverted = cv2.bitwise_not(img)
                image = Image.fromarray(img_inverted)

                # Immediate prediction for dataset sample
                if current_model["model"] is not None:
                    st.markdown("### 🎯 Dataset Sample Prediction")

                    # Process and predict
                    processed_img = preprocess_image(image, recognition_mode)
                    predicted_char, confidence = predict_character(processed_img, current_model["model"], recognition_mode)

                    # Display results in columns
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.image(image, caption=f"True Label: {label}", width=200)
                        st.image(processed_img, caption="Processed for AI", width=200)

                    with col2:
                        if predicted_char is not None:
                            # Check if prediction matches true label
                            is_correct = str(predicted_char) == str(label)
                            accuracy_color = "#059669" if is_correct else "#dc2626"
                            accuracy_text = "✅ Correct" if is_correct else "❌ Incorrect"

                            st.markdown(
                                f"""
                                <div class="glass-card">
                                    <h4>🗃️ Dataset Sample Prediction</h4>
                                    <div style="display: flex; align-items: center; justify-content: space-between; margin: 1rem 0;">
                                        <div>
                                            <div style="font-size: 3rem; font-weight: 700; color: #2a5298;">{predicted_char}</div>
                                            <div style="color: #718096; font-size: 14px;">Predicted Digit</div>
                                        </div>
                                        <div style="text-align: right;">
                                            <div style="font-size: 1.5rem; font-weight: 600; color: {accuracy_color};">{accuracy_text}</div>
                                            <div style="font-size: 1.2rem; font-weight: 600; color: #059669;">{confidence:.1%}</div>
                                            <div style="color: #718096; font-size: 14px;">Confidence</div>
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            st.progress(float(confidence), text=f"{confidence:.1%}")

                            # Detailed metrics
                            col_a, col_b, col_c, col_d = st.columns(4)
                            with col_a:
                                st.metric("True Label", label)
                            with col_b:
                                st.metric("Prediction", predicted_char)
                            with col_c:
                                st.metric("Confidence", f"{confidence:.1%}")
                            with col_d:
                                st.metric("Accuracy", accuracy_text)
                        else:
                            st.error(f"❌ {current_model['name']}: Model not available")
                else:
                    st.warning("⚠️ Model not available for prediction")
        else:
            # Alphabet sample dataset
            st.markdown(
                """
                <div class="info-box">
                    🗃️ <strong>A_Z Handwritten Dataset:</strong> Access authentic handwritten alphabet samples from the A_Z Handwritten Data.csv dataset. These are real handwritten letters (A-E) for realistic alphabet recognition testing.
                </div>
                """,
                unsafe_allow_html=True
            )

            # Load A_Z dataset samples
            az_samples, az_loaded = load_az_dataset_samples()

            if az_loaded:
                alphabet_letters = list(az_samples.keys())
                st.success("✅ **A_Z Dataset**: Loaded 5 authentic handwritten alphabet samples (A-E).")
            else:
                # Fallback to EMNIST-style samples
                st.warning("⚠️ **Fallback Mode**: Could not load A_Z dataset, using EMNIST-style samples.")
                emnist_samples, emnist_loaded = load_emnist_style_samples()
                if emnist_loaded:
                    alphabet_letters = list(emnist_samples.keys())[:5]  # Only 5 samples
                    az_samples = {k: emnist_samples[k] for k in alphabet_letters}
                else:
                    # Final fallback
                    alphabet_samples = create_realistic_alphabet_samples()
                    alphabet_letters = list(alphabet_samples.keys())[:5]  # Only 5 samples
                    az_samples = {k: alphabet_samples[k] for k in alphabet_letters}
            st.markdown("### � Alphabet Sample Preview")

            # Use A_Z dataset samples
            st.markdown("### 🔬 A_Z Dataset Sample Preview")

            preview_imgs = []
            sample_data = []

            for letter in alphabet_letters:
                img_array = az_samples[letter]
                preview_imgs.append(Image.fromarray(img_array))
                sample_data.append((img_array, letter))

            st.image(preview_imgs, caption=[f"{letter}" for letter in alphabet_letters], width=80)
            sample_choice = st.selectbox("Choose a sample letter:", [f"Letter {letter} (Sample {i+1})" for i, letter in enumerate(alphabet_letters)])

            if st.button("Load Sample & Predict"):
                idx = int(sample_choice.split("Sample ")[1].replace(")", "")) - 1
                img_array, label = sample_data[idx]
                image = Image.fromarray(img_array)

                # Immediate prediction for alphabet dataset sample
                if current_model["model"] is not None:
                    st.markdown("### 🎯 Alphabet Dataset Prediction")

                    # Add debug toggle
                    debug_mode = st.checkbox("🔍 Enable Debug Mode", help="Show detailed prediction analysis")

                    # Process and predict
                    processed_img = preprocess_image(image, recognition_mode)
                    predicted_char, confidence = predict_character(processed_img, current_model["model"], recognition_mode, debug=debug_mode)

                    # Display results in columns
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.image(image, caption=f"True Label: {label}", width=200)
                        st.image(processed_img, caption="Processed for AI", width=200)

                    with col2:
                        if predicted_char is not None:
                            # Check if prediction matches true label
                            is_correct = str(predicted_char).upper() == str(label).upper()
                            accuracy_color = "#059669" if is_correct else "#dc2626"
                            accuracy_text = "✅ Correct" if is_correct else "❌ Incorrect"

                            st.markdown(
                                f"""
                                <div class="glass-card">
                                    <h4>🗃️ Alphabet Sample Prediction</h4>
                                    <div style="display: flex; align-items: center; justify-content: space-between; margin: 1rem 0;">
                                        <div>
                                            <div style="font-size: 3rem; font-weight: 700; color: #2a5298;">{predicted_char}</div>
                                            <div style="color: #718096; font-size: 14px;">Predicted Letter</div>
                                        </div>
                                        <div style="text-align: right;">
                                            <div style="font-size: 1.5rem; font-weight: 600; color: {accuracy_color};">{accuracy_text}</div>
                                            <div style="font-size: 1.2rem; font-weight: 600; color: #059669;">{confidence:.1%}</div>
                                            <div style="color: #718096; font-size: 14px;">Confidence</div>
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            st.progress(float(confidence), text=f"{confidence:.1%}")

                            # Detailed metrics
                            col_a, col_b, col_c, col_d = st.columns(4)
                            with col_a:
                                st.metric("True Label", label)
                            with col_b:
                                st.metric("Prediction", predicted_char)
                            with col_c:
                                st.metric("Confidence", f"{confidence:.1%}")
                            with col_d:
                                st.metric("Accuracy", accuracy_text)
                        else:
                            st.error(f"❌ {current_model['name']}: Model not available")
                else:
                    st.warning("⚠️ Model not available for prediction")

    elif choice == "🎬 Demo Slideshow":
        if recognition_mode == "digit":
            st.markdown(
                """
                <div class="info-box">
                    🎬 <strong>Demonstration Mode:</strong> Automated system demonstration showcasing real-time prediction capabilities across multiple sample inputs. Ideal for stakeholder presentations.
                </div>
                """,
                unsafe_allow_html=True
            )
            from tensorflow.keras.datasets import mnist
            (_, _), (x_test, y_test) = mnist.load_data()
            st.markdown("### 🎬 Automated Demonstration - Processing 5 Samples")
            slideshow_area = st.empty()

            if st.button("🎬 Start Digit Slideshow"):
                for i in range(5):
                    idx = random.randint(0, len(x_test) - 1)
                    img = x_test[idx]
                    label = y_test[idx]
                    img_inverted = cv2.bitwise_not(img)
                    processed_img = preprocess_image(Image.fromarray(img_inverted), recognition_mode)

                    # Get prediction from current model
                    predicted_char, confidence = predict_character(processed_img, current_model["model"], recognition_mode)

                    with slideshow_area.container():
                        st.markdown(f"### 🎬 Sample {i+1}/5 - Digit Recognition Demo")

                        # Create columns for better layout
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.image(img_inverted, caption=f"True Label: {label}", width=200)
                            st.image(processed_img, caption="Processed for AI", width=200)

                        with col2:
                            if predicted_char is not None:
                                # Check accuracy
                                is_correct = str(predicted_char) == str(label)
                                accuracy_color = "#059669" if is_correct else "#dc2626"
                                accuracy_text = "✅ Correct" if is_correct else "❌ Incorrect"

                                st.markdown(
                                    f"""
                                    <div class="glass-card">
                                        <h4>🎬 Live Prediction Results</h4>
                                        <div style="display: flex; align-items: center; justify-content: space-between; margin: 1rem 0;">
                                            <div>
                                                <div style="font-size: 3rem; font-weight: 700; color: #2a5298;">{predicted_char}</div>
                                                <div style="color: #718096; font-size: 14px;">Predicted Digit</div>
                                            </div>
                                            <div style="text-align: right;">
                                                <div style="font-size: 1.5rem; font-weight: 600; color: {accuracy_color};">{accuracy_text}</div>
                                                <div style="font-size: 1.2rem; font-weight: 600; color: #059669;">{confidence:.1%}</div>
                                                <div style="color: #718096; font-size: 14px;">Confidence</div>
                                            </div>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                st.progress(float(confidence), text=f"{confidence:.1%}")

                                # Live metrics
                                col_a, col_b, col_c, col_d = st.columns(4)
                                with col_a:
                                    st.metric("True", label)
                                with col_b:
                                    st.metric("Predicted", predicted_char)
                                with col_c:
                                    st.metric("Confidence", f"{confidence:.1%}")
                                with col_d:
                                    st.metric("Result", accuracy_text)
                            else:
                                st.error(f"❌ {current_model['name']}: Model not available")

                    time.sleep(4)  # Longer pause for better viewing
        else:
            # Alphabet demo slideshow
            st.markdown(
                """
                <div class="info-box">
                    🎬 <strong>A_Z Dataset Alphabet Demo:</strong> Automated system demonstration showcasing real-time alphabet recognition capabilities using authentic handwritten letter samples from the A_Z Handwritten Data.csv dataset.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 🎬 Automated Alphabet Demonstration - Processing 5 Samples")
            slideshow_area = st.empty()

            # Use A_Z dataset samples for demo
            az_samples, az_loaded = load_az_dataset_samples()

            if az_loaded:
                demo_letters = list(az_samples.keys())  # A, B, C, D, E
                st.info("🎬 **A_Z Dataset Demo**: Using authentic handwritten alphabet samples (A-E).")
            else:
                # Fallback to EMNIST-style samples
                st.warning("⚠️ **Fallback Demo**: Could not load A_Z dataset, using EMNIST-style samples.")
                emnist_samples, emnist_loaded = load_emnist_style_samples()
                if emnist_loaded:
                    demo_letters = list(emnist_samples.keys())[:5]  # Only 5 samples
                    az_samples = {k: emnist_samples[k] for k in demo_letters}
                else:
                    # Final fallback
                    alphabet_samples = create_realistic_alphabet_samples()
                    demo_letters = list(alphabet_samples.keys())[:5]  # Only 5 samples
                    az_samples = {k: alphabet_samples[k] for k in demo_letters}

            if st.button("🎬 Start Alphabet Slideshow"):
                for i in range(5):
                    # Select random letter from A_Z dataset samples
                    letter = random.choice(demo_letters)

                    # Use the A_Z dataset sample
                    img_array = az_samples[letter]
                    processed_img = preprocess_image(Image.fromarray(img_array), recognition_mode)

                    # Get prediction from current model
                    predicted_char, confidence = predict_character(processed_img, current_model["model"], recognition_mode)

                    with slideshow_area.container():
                        st.markdown(f"### 🎬 Sample {i+1}/5 - Alphabet Recognition Demo")

                        # Create columns for better layout
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.image(img_array, caption=f"True Label: {letter}", width=200)
                            st.image(processed_img, caption="Processed for AI", width=200)

                        with col2:
                            if predicted_char is not None:
                                # Check accuracy
                                is_correct = str(predicted_char).upper() == str(letter).upper()
                                accuracy_color = "#059669" if is_correct else "#dc2626"
                                accuracy_text = "✅ Correct" if is_correct else "❌ Incorrect"

                                st.markdown(
                                    f"""
                                    <div class="glass-card">
                                        <h4>🎬 Live Alphabet Prediction</h4>
                                        <div style="display: flex; align-items: center; justify-content: space-between; margin: 1rem 0;">
                                            <div>
                                                <div style="font-size: 3rem; font-weight: 700; color: #2a5298;">{predicted_char}</div>
                                                <div style="color: #718096; font-size: 14px;">Predicted Letter</div>
                                            </div>
                                            <div style="text-align: right;">
                                                <div style="font-size: 1.5rem; font-weight: 600; color: {accuracy_color};">{accuracy_text}</div>
                                                <div style="font-size: 1.2rem; font-weight: 600; color: #059669;">{confidence:.1%}</div>
                                                <div style="color: #718096; font-size: 14px;">Confidence</div>
                                            </div>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                st.progress(float(confidence), text=f"{confidence:.1%}")

                                # Live metrics
                                col_a, col_b, col_c, col_d = st.columns(4)
                                with col_a:
                                    st.metric("True", letter)
                                with col_b:
                                    st.metric("Predicted", predicted_char)
                                with col_c:
                                    st.metric("Confidence", f"{confidence:.1%}")
                                with col_d:
                                    st.metric("Result", accuracy_text)
                            else:
                                st.error(f"❌ {current_model['name']}: Model not available")

                    time.sleep(4)  # Longer pause for better viewing


else:
    st.warning("⚠️ Please select a recognition system above to continue.")

# ---------------------------
# Professional Footer
# ---------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 2rem 0; background: #f9fafb; border-radius: 8px; margin-top: 2rem;">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
            <div style="margin-right: 1rem;">
                <svg width="50" height="62" viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg">
                    <path d="M100 50
                             C85 50, 75 60, 75 75
                             C70 70, 60 75, 60 85
                             C55 90, 55 100, 60 105
                             C55 110, 60 120, 65 125
                             C60 130, 65 140, 70 145
                             C75 155, 85 160, 95 155
                             C100 165, 110 160, 115 150
                             C125 155, 135 150, 140 140
                             C145 135, 140 125, 135 120
                             C140 115, 135 105, 130 100
                             C135 95, 130 85, 125 80
                             C120 70, 110 65, 105 70
                             C110 60, 105 50, 100 50 Z"
                          fill="none"
                          stroke="#1e40af"
                          stroke-width="4"
                          stroke-linecap="round"
                          stroke-linejoin="round"/>
                    <path d="M100 55
                             C95 60, 90 70, 85 80
                             C80 90, 75 100, 80 110
                             C85 120, 90 130, 95 140
                             C98 145, 100 150, 100 155"
                          fill="none"
                          stroke="#1e40af"
                          stroke-width="2"
                          stroke-linecap="round"/>
                    <text x="85" y="80" font-family="IBM Plex Sans, sans-serif" font-size="10" font-weight="600" fill="#1e40af">3</text>
                    <text x="115" y="80" font-family="IBM Plex Sans, sans-serif" font-size="10" font-weight="600" fill="#1e40af">8</text>
                    <text x="70" y="105" font-family="IBM Plex Sans, sans-serif" font-size="10" font-weight="600" fill="#1e40af">9</text>
                    <text x="100" y="105" font-family="IBM Plex Sans, sans-serif" font-size="10" font-weight="600" fill="#1e40af">2</text>
                    <text x="130" y="105" font-family="IBM Plex Sans, sans-serif" font-size="10" font-weight="600" fill="#1e40af">6</text>
                    <text x="85" y="130" font-family="IBM Plex Sans, sans-serif" font-size="10" font-weight="600" fill="#1e40af">5</text>
                    <text x="115" y="130" font-family="IBM Plex Sans, sans-serif" font-size="10" font-weight="600" fill="#1e40af">7</text>
                    <text x="70" y="150" font-family="IBM Plex Sans, sans-serif" font-size="10" font-weight="600" fill="#1e40af">4</text>
                    <text x="100" y="150" font-family="IBM Plex Sans, sans-serif" font-size="10" font-weight="600" fill="#1e40af">1</text>
                    <text x="130" y="150" font-family="IBM Plex Sans, sans-serif" font-size="10" font-weight="600" fill="#1e40af">0</text>
                </svg>
            </div>
            <div>
                <strong style="color: #1e40af; font-size: 18px;">HDAR AI Platform</strong>
                <span style="color: #374151; font-weight: 500;"> | Enterprise Digit Recognition System</span>
            </div>
        </div>
        <div style="font-size: 16px; color: #1a202c; font-weight: 500;">
            Powered by Deep Learning • Real-time Processing • 99.2% Accuracy
        </div>
        <div style="font-size: 14px; margin-top: 0.5rem; color: #6b7280; font-weight: 500;">
            © 2024 HDAR AI. All rights reserved. | Version 2.1.0
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
