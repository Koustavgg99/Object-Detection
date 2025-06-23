import streamlit as st
import cv2
from datetime import datetime
from transformers import pipeline
from PIL import Image
import numpy as np
import os
import torch
import warnings
warnings.filterwarnings("ignore")

def initialize_classifier():
    """Initialize the ConvNeXt image classification model"""
    model_name = "facebook/convnext-tiny-224"
    try:
        
        classifier = pipeline(
            "image-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        
        return classifier
    except Exception as e:
        st.error(f"Failed to load {model_name}: {str(e)}")
        raise Exception(f"Failed to load the required model: {model_name}")

# Initialize the classifier

try:
    classifier = initialize_classifier()
except Exception as e:
    st.error(f"Error: {e}")
    st.error("Please check your internet connection and try again.")
    st.stop()

# Create output directory
output_dir = "biscuit_packets"
os.makedirs(output_dir, exist_ok=True)

def is_biscuit_packet(labels):
    """Check if any of the predicted labels indicate a biscuit packet/container"""
    # Refined keywords focusing on biscuits, excluding generic packaging terms
    biscuit_keywords = [
        "biscuit","packet","creme" "cookie", "cracker", "wafer", "shortbread",
        
        "gingerbread", "digestive", "oreo", "biscotti", "tea biscuit",
        "macaroon", "sandwich biscuit"
    ]
    
    # Terms to exclude (e.g., candies, unrelated snacks)
    exclude_keywords = [
        "candy", "sweet", "mint", "gum", "lollipop",
        "tic tac", "confectionery", "toffee","cake", "chocolate"
    ]
    
    st.write("**Top predictions:**")
    best_match = None
    best_score = 0
    
    for i, label_info in enumerate(labels[:10]):  # Check top 10 predictions
        label_text = label_info['label'].lower()
        confidence = label_info['score']
        st.write(f"{i+1}. {label_info['label']}: {confidence:.3f}")
        
        # Check for biscuit-specific keywords
        for keyword in biscuit_keywords:
            if keyword in label_text:
                # Ensure the label doesn't contain excluded terms
                if not any(exclude in label_text for exclude in exclude_keywords):
                    if confidence > best_score:
                        best_match = label_info['label']
                        best_score = confidence
    
    # Increased confidence threshold for stricter detection
    min_confidence = 0.2  # Raised from 0.1 to reduce false positives
    
    if best_match and best_score >= min_confidence:
        return True, best_match, best_score
    return False, None, 0

def analyze_image(img):
    """Analyze the provided image"""
    try:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image if it's too large
        if img.size[0] > 800 or img.size[1] > 600:
            img.thumbnail((800, 600), Image.Resampling.LANCZOS)
        
        st.write("Analyzing image...")
        results = classifier(img)
        
        is_biscuit_detected, detected_label, confidence = is_biscuit_packet(results)
        
        if is_biscuit_detected:
            filename = f"biscuit_packet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            output_path = os.path.join(output_dir, filename)
            img.save(output_path, 'JPEG', quality=95)
            
            st.success("**BISCUIT PACKET DETECTED!**")
            st.write(f"Detected as: {detected_label}")
            st.write(f"Confidence: {confidence:.1%}")
            
            return True
        else:
            st.warning("**NO BISCUIT PACKET DETECTED**")
            return False
            
    except Exception as e:
        st.error(f"Error during image analysis: {str(e)}")
        return False

def main():
    st.title("Biscuit Packet Detection ")
    # Initialize session state for mode and analysis status
    if 'mode' not in st.session_state:
        st.session_state['mode'] = None
    if 'analysis_done' not in st.session_state:
        st.session_state['analysis_done'] = False

    
    # If analysis is done, show DONE button
    
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Capture Photo"):
            st.session_state['mode'] = 'capture'
            st.session_state['analysis_done'] = False
    
    with col2:
        if st.button("Upload from Media"):
            st.session_state['mode'] = 'upload'
            st.session_state['analysis_done'] = False
    
    # Handle capture mode
    if st.session_state['mode'] == 'capture':
        st.subheader("Camera Capture")
        st.write("Use your device's camera to capture a biscuit packet image.")
        camera_image = st.camera_input("Take a photo")
        
        if camera_image:
            img = Image.open(camera_image)
            st.image(img, caption="Captured Image", use_container_width=True)
            analyze_image(img)
            st.session_state['analysis_done'] = True
    
    # Handle upload mode
    if st.session_state['mode'] == 'upload':
        st.subheader("Upload Image")
        st.write("Upload an image of a biscuit packet from your device.")
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'])
        
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            analyze_image(img)
            st.session_state['analysis_done'] = True

if __name__ == "__main__":
    main()