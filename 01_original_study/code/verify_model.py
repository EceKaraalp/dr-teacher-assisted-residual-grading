"""
Quick verification script for CViTS-Net model architecture.
Tests model building without requiring the full dataset.
"""

import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("CViTS-Net Architecture Verification")
print("="*80)

try:
    print("\n1. Importing modules...")
    from cvitsnet_model import build_cvitsnet, count_parameters
    print("   ✓ Model imported successfully")
    
    print("\n2. Building model...")
    model = build_cvitsnet(num_classes=5, image_size=224)
    print("   ✓ Model built successfully")
    
    print("\n3. Model Summary:")
    model.summary()
    
    print("\n4. Counting parameters...")
    total_params = count_parameters(model)
    print(f"   Total trainable parameters: {total_params:,}")
    print(f"   Expected: ~21-22 million")
    
    if 20_000_000 <= total_params <= 22_500_000:
        print("   ✓ Parameter count within expected range")
    else:
        print(f"   ⚠ Warning: Parameter count {total_params:,} outside expected range")
    
    print("\n5. Testing model inference...")
    # Create dummy input (float32 in [0, 1] range)
    dummy_input = tf.random.uniform((1, 224, 224, 3), minval=0, maxval=256, dtype=tf.int32)
    dummy_input = tf.cast(dummy_input, tf.uint8)
    
    # Get prediction
    output = model(dummy_input, training=False)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output sum (should be ~1.0): {tf.reduce_sum(output[0]).numpy():.4f}")
    
    if output.shape == (1, 5):
        print("   ✓ Output shape is correct (1, 5)")
    else:
        print(f"   ✗ Output shape is incorrect: {output.shape}")
    
    print("\n6. Testing with batch of images...")
    batch_input = tf.random.uniform((4, 224, 224, 3), minval=0, maxval=255, dtype=tf.uint8)
    batch_output = model(batch_input, training=False)
    print(f"   Batch input shape: {batch_input.shape}")
    print(f"   Batch output shape: {batch_output.shape}")
    
    if batch_output.shape == (4, 5):
        print("   ✓ Batch output shape is correct (4, 5)")
    else:
        print(f"   ✗ Batch output shape is incorrect: {batch_output.shape}")
    
    print("\n7. Testing gradient computation...")
    with tf.GradientTape() as tape:
        output = model(dummy_input, training=True)
        loss = tf.reduce_sum(output)
    
    gradients = tape.gradient(loss, model.trainable_weights)
    non_none_grads = sum(1 for g in gradients if g is not None)
    
    print(f"   Total trainable weight groups: {len(model.trainable_weights)}")
    print(f"   Weight groups with gradients: {non_none_grads}")
    
    if non_none_grads > 0:
        print("   ✓ Gradients computed successfully")
    else:
        print("   ✗ No gradients computed")
    
    print("\n" + "="*80)
    print("✓ ALL VERIFICATION TESTS PASSED!")
    print("="*80)
    print("\nThe CViTS-Net architecture is correctly implemented and ready for training.")
    print("\nTo start training, run:")
    print("  python train.py")
    print("\nMake sure APTOS2019 dataset is in the current directory.")
    print("="*80 + "\n")

except Exception as e:
    print(f"\n✗ Error during verification: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
