# src/cnn_architecture.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_eye_disease_cnn(input_shape=(224, 224, 1), num_classes=4):
    """
    Constructs a structural Convolutional Neural Network (CNN) architecture
    tailored for multi-class retinal OCT eye disease classification.
    """
    print(f"[INFO] Initializing CNN Architecture with input shape: {input_shape}...")
    
    model = models.Sequential(name="Retinal_Disease_Classifier")
    
    # Explicit modern Input layer to satisfy Keras 3 standard specifications
    model.add(layers.Input(shape=input_shape))
    
    # 1. FIRST CONVOLUTIONAL BLOCK
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # 2. SECOND CONVOLUTIONAL BLOCK
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # 3. TRANSITION TO CLASSIFIER (FLATTENING)
    model.add(layers.Flatten())
    
    # 4. DENSELY CONNECTED INTERPRETATION LAYER
    model.add(layers.Dense(units=128, activation='relu'))
    
    # 5. FINAL MULTI-CLASS OUTPUT LAYER
    model.add(layers.Dense(units=num_classes, activation='softmax'))
    
    return model

if __name__ == "__main__":
    classifier = build_eye_disease_cnn()
    classifier.summary()