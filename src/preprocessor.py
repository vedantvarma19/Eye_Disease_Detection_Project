# src/preprocessor.py

import cv2
import numpy as np
import os

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads a raw medical scan, standardizes color spaces, resizes dimensions,
    and applies clinical CLAHE contrast enhancement.
    """
    # Absolute path expansion to help debug path tracking issues
    abs_path = os.path.abspath(image_path)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] File does not exist at path: {abs_path}")
        
    print(f"[INFO] Attempting to load image from: {abs_path}")
    
    # 1. Load image via OpenCV
    raw_bgr = cv2.imread(image_path)
    
    # NEW SAFETY CHECK: Catch empty image loading issues immediately
    if raw_bgr is None:
        raise ValueError(
            f"[ERROR] OpenCV could not decode the image file at '{abs_path}'.\n"
            f"Please verify that the file is not corrupted and is a valid image format."
        )
    
    # 2. Convert to Grayscale
    gray_img = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply CLAHE enhancement
    print("[INFO] Applying CLAHE enhancement...")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray_img)
    
    # 4. Resize uniformly
    print(f"[INFO] Resizing image to fixed dimensions: {target_size}")
    resized_img = cv2.resize(enhanced_img, target_size, interpolation=cv2.INTER_AREA)
    
    # 5. Expand dimensions to create a mock batch tensor shape: (1, Height, Width, 1)
    final_tensor = np.expand_dims(resized_img, axis=(0, -1))
    
    # 6. Normalize to [0.0, 1.0] float range
    normalized_tensor = final_tensor / 255.0
    
    return raw_bgr, resized_img, normalized_tensor

if __name__ == "__main__":
    target_path = os.path.join("sample_data", "test_eye.jpg")
    
    try:
        raw, enhanced, tensor = preprocess_image(target_path)
        
        print("\n========= PREPROCESSING PIPELINE SUCCESS =========")
        print(f"Raw Image Array Shape        : {raw.shape}")
        print(f"Enhanced 2D Matrix Shape     : {enhanced.shape}")
        print(f"Final Model Input Tensor Shape: {tensor.shape}")
        print(f"Tensor Value Bounds          : Min={tensor.min()}, Max={tensor.max()}")
        print("==================================================\n")
        
    except Exception as e:
        print(e)