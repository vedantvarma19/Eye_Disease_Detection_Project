# src/app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# Import your core custom modules built throughout this project
from preprocessor import preprocess_image
from recommendation import get_clinical_guidance

# Configure page metadata layout properties
st.set_page_config(
    page_title="Retinal Eye Disease Detector",
    page_icon="👁️",
    layout="wide"
)

@st.cache_resource
def load_cached_model():
    """
    st.cache_resource ensures the model is loaded into memory only ONCE 
    when the app boots up, preventing slow performance on subsequent page reloads.
    """
    model_path = os.path.join("models", "eye_disease_model.keras")
    if not os.path.exists(model_path):
        # Fallback to create a structural template if user hasn't fully trained one
        from evaluator import save_trained_model
        from train import compile_and_initialize_training
        model = compile_and_initialize_training()
        save_trained_model(model)
        
    return tf.keras.models.load_model(model_path)

# Initialize application dependencies
model = load_cached_model()
class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]

# Render dashboard headers
st.title("👁️ Retinal Eye Disease Detection System")
st.markdown("---")

# Layout dashboard into two equal structural columns (Left: Controls, Right: Outputs)
col1, col2 = st.columns(2)

with col1:
    st.header("📥 Upload Retinal Scan")
    uploaded_file = st.file_uploader(
        "Choose an Optical Coherence Tomography (OCT) image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Create a transient temporary file path to pass cleanly to OpenCV
        temp_path = os.path.join("sample_data", "temp_upload.jpg")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success("File uploaded successfully! Click predict below.")
        predict_btn = st.button("🚀 Run System Diagnostics")

with col2:
    st.header("📊 Analytical Output Workspace")
    
    if uploaded_file is not None:
        # Run our custom multi-stage OpenCV preprocessing engine
        raw_img, enhanced_img, input_tensor = preprocess_image(temp_path)
        
        # Display side-by-side verification previews for the user
        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            st.image(uploaded_file, caption="User Uploaded Scan", use_container_width=True)
        with preview_col2:
            st.image(enhanced_img, caption="CLAHE Enhanced View", use_container_width=True)
            
        if 'predict_btn' in locals() and predict_btn:
            # 1. Run network inference forward pass
            probabilities = model.predict(input_tensor)[0]
            
            # 2. Extract highest probability ID and score
            predicted_id = np.argmax(probabilities)
            confidence = probabilities[predicted_id]
            
            # 3. Pull diagnostic notes from rules engine
            guidance = get_clinical_guidance(predicted_id)
            
            # Render predictions prominently
            st.markdown("---")
            st.subheader(f"System Assessment: **{guidance['name']}**")
            st.metric(label="Prediction Confidence Score", value=f"{confidence * 100:.2f}%")
            
            # Render clinical guidance info panels
            if predicted_id == 3:
                st.info(f"**Pathology Breakdown:** {guidance['description']}")
            else:
                st.error(f"**Pathology Breakdown:** {guidance['description']}")
                
            st.warning(f"**Clinical Urgency Profile:** {guidance['urgency']}")
            
            st.markdown("### ⚡ Immediate Care Precautions:")
            for precaution in guidance['precautions']:
                st.markdown(f"- {precaution}")
    else:
        st.info("Awaiting input retinal image file to initialize diagnostic analytics pipelines.")