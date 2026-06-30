# src/single_neuron.py

import numpy as np

def sigmoid_activation(z):
    """
    The mathematical Sigmoid activation function.
    Squashes any real-valued number into a probability range between 0 and 1.
    """
    return 1.0 / (1.0 + np.exp(-z))

def simulate_single_neuron(inputs, weights, bias):
    """
    Computes the forward pass of a single artificial neuron.
    Formula: Output = Sigmoid( Dot_Product(Inputs, Weights) + Bias )
    """
    # 1. Compute the dot product of inputs and weights, then add bias
    # In Java, this would be a long loop multiplying array items.
    # In NumPy, np.dot does it instantly in optimized C code.
    raw_sum = np.dot(inputs, weights) + bias
    print(f"[MATHEMATICS] Weighted Summation (z) = {raw_sum}")
    
    # 2. Pass the raw summation through our non-linear activation function
    output_probability = sigmoid_activation(raw_sum)
    return output_probability

if __name__ == "__main__":
    print("========= ARTIFICIAL NEURON SIMULATION =========")
    
    # Imagine we extract 3 specific preprocessed pixel features from an eye scan
    # Input features scaled between 0.0 and 1.0
    mock_pixel_inputs = np.array([0.8, 0.2, 0.5])
    
    # Initialize mock weights (learned strengths for each pixel feature)
    mock_weights = np.array([2.5, -1.2, 0.5])
    
    # Initialize a bias value
    mock_bias = -1.0
    
    print(f"1. Input Values (X) : {mock_pixel_inputs}")
    print(f"2. Connection Weights (W): {mock_weights}")
    print(f"3. Neuron Bias (b)       : {mock_bias}")
    
    # Compute output
    prediction = simulate_single_neuron(mock_pixel_inputs, mock_weights, mock_bias)
    
    print(f"4. Neuron Prediction Output: {prediction:.4f}")
    
    if prediction >= 0.5:
        print("   -> Decision: Disease Detected! (High Probability)")
    else:
        print("   -> Decision: Normal Healthy Retina (Low Probability)")
    print("================================================\n")