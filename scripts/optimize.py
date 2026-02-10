"""
TFLite conversion and model optimization for edge deployment.
"""

import tensorflow as tf
import os

def convert_keras_to_tflite(model_path, output_path='../models/plant_model_v1.tflite'):
    """
    Convert Keras model to TFLite format with optimizations.
    
    Args:
        model_path: Path to the trained model (.keras format)
        output_path: Path to save the converted TFLite model
    """
    print(f"Loading model from {model_path}...")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return False
    
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)
    
    # Create converter from Keras model
    print("Creating TFLiteConverter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    print("Applying optimizations...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Optional: Set target ops for INT8 quantization (more aggressive)
    # Uncomment for maximum compression on edge devices
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    # ]
    # converter.inference_input_type = tf.int8
    # converter.inference_output_type = tf.int8
    
    # Convert model
    print("Converting model to TFLite format...")
    tflite_model = converter.convert()
    
    # Save TFLite model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    original_size = os.path.getsize(model_path)
    tflite_size = os.path.getsize(output_path)
    compression_ratio = (1 - tflite_size / original_size) * 100
    
    print(f"\n✓ Conversion successful!")
    print(f"Original model size: {original_size / 1024 / 1024:.2f} MB")
    print(f"TFLite model size:   {tflite_size / 1024 / 1024:.2f} MB")
    print(f"Compression ratio:   {compression_ratio:.1f}%")
    print(f"Model saved to:      {output_path}")
    
    return True

def test_tflite_model(tflite_path, test_image_path=None):
    """
    Test TFLite model and verify it works.
    
    Args:
        tflite_path: Path to the TFLite model
        test_image_path: Optional path to test image
    """
    print(f"\nLoading TFLite model from {tflite_path}...")
    
    # Create interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✓ Model loaded successfully!")
    print(f"\nInput shape:  {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Input dtype:  {input_details[0]['dtype']}")
    print(f"Output dtype: {output_details[0]['dtype']}")
    
    return interpreter, input_details, output_details

if __name__ == "__main__":
    # Convert Keras model to TFLite
    model_file = '../models/plant_model_v1.keras'
    tflite_file = '../models/plant_model_v1.tflite'
    
    if convert_keras_to_tflite(model_file, tflite_file):
        # Test the converted model
        test_tflite_model(tflite_file)
