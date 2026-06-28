# src/data_loader.py

import numpy as np

def generate_mock_image_batch(batch_size=4, height=224, width=224, channels=3):
    """
    Simulates loading a batch of preprocessed retinal OCT scan images.
    Returns a 4D NumPy array representing (Batch, Height, Width, Channels).
    """
    print(f"[INFO] Generating a mock batch of {batch_size} synthetic retinal scans...")
    
    # 8-bit unsigned integers (0-255) to mimic a raw digital image file
    mock_batch = np.random.randint(
        low=0, 
        high=256, 
        size=(batch_size, height, width, channels), 
        dtype=np.uint8
    )
    return mock_batch

def normalize_images(tensor):
    """
    Scales pixel values from integer range [0, 255] to floating-point range [0.0, 1.0].
    This keeps network gradients stable during backpropagation.
    """
    print("[INFO] Normalizing image tensor values to range [0.0, 1.0]...")
    
    # Dividing an integer array by a float (255.0) implicitly casts the 
    # entire resulting array into floating-point numbers (float64 or float32).
    normalized_tensor = tensor / 255.0
    return normalized_tensor

def inspect_tensor_properties(tensor, label="RAW TENSOR"):
    """
    Examines the structural properties of a data tensor.
    """
    print(f"\n========= TENSOR INTEGRITY REPORT ({label}) =========")
    print(f"1. Data Type (dtype) : {tensor.dtype}")
    print(f"2. Number of Dimensions: {tensor.ndim}")
    print(f"3. Tensor Shape        : {tensor.shape}")
    print(f"4. Min Pixel Value     : {np.min(tensor)}")
    print(f"5. Max Pixel Value     : {np.max(tensor)}")
    print("=======================================================\n")

if __name__ == "__main__":
    # 1. Generate the raw synthetic images
    raw_images = generate_mock_image_batch()
    inspect_tensor_properties(raw_images, label="RAW IMAGES")
    
    # 2. Normalize the images for the neural network
    normalized_images = normalize_images(raw_images)
    inspect_tensor_properties(normalized_images, label="NORMALIZED IMAGES")