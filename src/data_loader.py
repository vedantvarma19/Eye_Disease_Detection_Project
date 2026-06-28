# src/data_loader.py

import numpy as np

def generate_mock_image_batch(batch_size=4, height=224, width=224, channels=3):
    """
    Simulates loading a batch of preprocessed retinal OCT scan images.
    Returns a 4D NumPy array representing (Batch, Height, Width, Channels).
    """
    print(f"[INFO] Generating a mock batch of {batch_size} synthetic retinal scans...")
    
    # np.random.randint generates random integers within a specific range.
    # We choose 0 to 256 because pixel values are 8-bit unsigned integers (0-255).
    # 'dtype=np.uint8' stands for Unsigned Integer 8-bit, optimizing memory layout.
    mock_batch = np.random.randint(
        low=0, 
        high=256, 
        size=(batch_size, height, width, channels), 
        dtype=np.uint8
    )
    
    return mock_batch

def inspect_tensor_properties(tensor):
    """
    Examines the structural properties of a data tensor, simulating 
    a senior engineer checking data integrity shapes before training.
    """
    print("\n========= TENSOR INTEGRITY REPORT =========")
    print(f"1. Data Type (dtype) : {tensor.dtype}")
    print(f"2. Number of Dimensions: {tensor.ndim}")
    print(f"3. Tensor Shape        : {tensor.shape}")
    print(f"   -> Batch Size (Images) : {tensor.shape[0]}")
    print(f"   -> Image Height        : {tensor.shape[1]} pixels")
    print(f"   -> Image Width         : {tensor.shape[2]} pixels")
    print(f"   -> Color Channels      : {tensor.shape[3]} (RGB)")
    print(f"4. Total Elements (Pix): {tensor.size}")
    print("===========================================\n")

if __name__ == "__main__":
    # Simulate loading our first batch of images
    image_batch = generate_mock_image_batch()
    inspect_tensor_properties(image_batch)