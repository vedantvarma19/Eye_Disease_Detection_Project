# src/train.py

import tensorflow as tf
import numpy as np
from cnn_architecture import build_eye_disease_cnn
from data_loader import generate_mock_image_batch

def compile_and_initialize_training():
    """
    Instantiates the CNN architecture, configures optimization algorithms,
    and sets up a safe compilation execution state.
    """
    print("[INFO] Constructing fresh model instance...")
    # Initialize our 224x224 grayscale model structure
    model = build_eye_disease_cnn(input_shape=(224, 224, 1), num_classes=4)
    
    print("[INFO] Compiling model with Optimization engines...")
    # .compile links the architecture with optimization mathematical components
    model.compile(
        # Adam dynamically optimizes learning speeds per parameter
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        
        # SparseCategoricalCrossentropy is ideal when integer labels (0, 1, 2, 3) 
        # map directly to our multi-class layout without one-hot encoding overhead.
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        
        # Metrics to observe performance output configurations
        metrics=['accuracy']
    )
    
    print("[SUCCESS] Model successfully compiled and prepared for training loops.")
    return model

if __name__ == "__main__":
    print("========= TRAINING ENGINE INITIALIZATION =========")
    # 1. Compile the network
    compiled_model = compile_and_initialize_training()
    
    # 2. Let's dry-run test a forward-pass optimization step using synthetic data
    # Create 4 synthetic single-channel grayscale images
    mock_x = generate_mock_image_batch(batch_size=4, height=224, width=224, channels=1)
    
    # Normalize inputs to scale range to float values between 0.0 and 1.0
    mock_x = mock_x / 255.0
    
    # Generate 4 mock integer labels mapping to our categories (0: CNV, 1: DME, 2: DRUSEN, 3: NORMAL)
    mock_y = np.array([0, 1, 2, 3])
    
    print("\n[INFO] Running 1 structural dry-run training epoch to verify execution pipelines...")
    # model.fit executes the actual forward pass, loss derivation, and backward weight tuning
    history = compiled_model.fit(mock_x, mock_y, epochs=1, batch_size=4)
    print("==================================================\n")