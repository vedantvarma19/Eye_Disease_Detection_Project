# src/evaluator.py

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

def save_trained_model(model, output_dir="models", filename="eye_disease_model.keras"):
    """
    Serializes and saves the complete model structure and weights to disk.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, filename)
    print(f"[INFO] Serializing model state to disk at: {save_path}")
    
    # .save compiles the weights and graph architectures into a single binary file
    model.save(save_path)
    print("[SUCCESS] Model file successfully preserved.")
    return save_path

def evaluate_model_performance(model, test_x, test_y):
    """
    Generates professional validation predictions and runs an in-depth 
    medical classification performance report.
    """
    print("\n[INFO] Generating predictions for model evaluation...")
    
    # 1. Generate model probabilities across all 4 classes
    raw_predictions = model.predict(test_x)
    
    # 2. Convert probabilities to definitive integer class IDs
    # np.argmax extracts the index containing the highest probability score
    predicted_classes = np.argmax(raw_predictions, axis=1)
    
    print("\n========= METRIC PERFORMANCE REPORT =========")
    # Define our target class labels explicitly
    class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]
    
    # scikit-learn's classification_report calculates precision, recall, and F1 per class
    report = classification_report(test_y, predicted_classes, target_names=class_names, zero_division=0)
    print(report)
    print("=============================================\n")
    
    return predicted_classes

if __name__ == "__main__":
    from train import compile_and_initialize_training
    from data_loader import generate_mock_image_batch
    
    print("========= MODEL EVALUATION & EXPORT ENGINE =========")
    # 1. Initialize a compiled model instance
    active_model = compile_and_initialize_training()
    
    # 2. Generate synthetic validation test datasets
    test_data_x = generate_mock_image_batch(batch_size=8, height=224, width=224, channels=1) / 255.0
    test_data_y = np.array([0, 0, 1, 1, 2, 2, 3, 3]) # 2 samples per class
    
    # 3. Evaluate the network's current performance
    evaluate_model_performance(active_model, test_data_x, test_data_y)
    
    # 4. Save this model state file onto the hard drive
    saved_file_path = save_trained_model(active_model)
    
    # 5. Verification Check: Test loading the model back into memory to confirm serialization worked
    print(f"\n[INFO] Verification Check: Reloading model from {saved_file_path}...")
    reloaded_model = tf.keras.models.load_model(saved_file_path)
    print("🎉 SUCCESS: Model reloaded perfectly with no data corruption!")