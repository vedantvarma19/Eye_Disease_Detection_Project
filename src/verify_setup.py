# src/verify_setup.py

import tensorflow as tf
import numpy as np
import cv2

def run_system_check():
    print("=========================================")
    # 1. Check Python environments via library bindings
    print(f"[STATUS] TensorFlow Version: {tf.__version__}")
    print(f"[STATUS] NumPy Version: {np.__version__}")
    print(f"[STATUS] OpenCV Version: {cv2.__version__}")
    print("=========================================")
    
    # 2. Hardware Accel Check: Query the host OS via TensorFlow's config API
    # list_physical_devices scans system buses for matching hardware signatures
    physical_gpus = tf.config.list_physical_devices('GPU')
    
    if len(physical_gpus) > 0:
        print(f"🎉 SUCCESS: GPU Available! Detected {len(physical_gpus)} hardware accelerator(s).")
        for i, gpu in enumerate(physical_gpus):
            print(f"   -> GPU {i}: {gpu.name}")
    else:
        print("💻 STATUS: Running on CPU only.")
        print("   (Note: This is completely normal if your system lacks a dedicated NVidia GPU.")
        print("    Our pipeline is built to train seamlessly on CPU as well!)")
    print("=========================================")

# This block ensures the code only runs when executing this file directly
if __name__ == "__main__":
    run_system_check()