import torch
import time
from modified_yolov8 import create_enhanced_yolov8, save_model_with_yaml, load_model_with_yaml

def visualize_model_structure(model):
    """Print the model structure including our custom layers."""
    print("\n=== Model Structure ===")
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Look for our custom modules
    if hasattr(model, 'custom_modules'):
        print("\nCustom modules:")
        for name, module in model.custom_modules.items():
            print(f"  - {name}: {type(module).__name__}")
    else:
        print("\nNo custom modules found")
    
    # Print insertion points if available
    if hasattr(model, 'insertion_points'):
        print("\nInsertion points:")
        for point in model.insertion_points:
            print(f"  - {point}")
    
    # Check model attributes
    if hasattr(model, 'channel_sizes'):
        print("\nChannel sizes:")
        for layer, channels in model.channel_sizes.items():
            print(f"  - {layer}: {channels}")
    
    # Test forward pass
    print("\nTesting forward pass with dummy input...")
    try:
        dummy_input = torch.randn(1, 3, 640, 640)
        start_time = time.time()
        with torch.no_grad():
            output = model(dummy_input)
        end_time = time.time()
        
        if isinstance(output, torch.Tensor):
            print(f"Forward pass successful! Output shape: {output.shape}")
        else:
            print(f"Forward pass successful! Output type: {type(output)}")
            if hasattr(output, 'shape'):
                print(f"Output shape: {output.shape}")
        
        print(f"Inference time: {(end_time - start_time)*1000:.2f} ms")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

def test_detection(model):
    """Test if the model can actually perform detection."""
    try:
        print("\nTesting detection capabilities...")
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model(dummy_input)
        
        # Verify output format - should match YOLOv8 detection output format
        print(f"Detection output type: {type(output)}")
        
        # If output is a list, it might be predictions
        if isinstance(output, list):
            print(f"Output list length: {len(output)}")
            if len(output) > 0:
                print(f"First item type: {type(output[0])}")
                if hasattr(output[0], 'shape'):
                    print(f"First item shape: {output[0].shape}")
        
        # If output is a tensor, check if it has the right dimensions for detection
        elif isinstance(output, torch.Tensor):
            print(f"Output tensor shape: {output.shape}")
            # Typical detection output has shape [batch, anchors, xywh+conf+classes]
            if len(output.shape) >= 2 and output.shape[-1] >= 5:
                print("Output format looks like valid detections")
            else:
                print("Output tensor doesn't have expected detection format")
    except Exception as e:
        print(f"Error testing detection: {e}")
        import traceback
        traceback.print_exc()

def test_save_load(model):
    """Test saving and loading the enhanced model."""
    print("\n=== Testing Save and Load Functionality ===")
    try:
        # Save the model
        model_path = "enhanced_yolov8_test.pt"
        save_model_with_yaml(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Load the model
        loaded_model = load_model_with_yaml(model_path)
        print("Model loaded successfully")
        
        # Test forward pass of loaded model
        print("\nTesting forward pass of loaded model...")
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = loaded_model(dummy_input)
        
        print("Forward pass of loaded model successful!")
        
        # Check custom modules in loaded model
        if hasattr(loaded_model, 'custom_modules'):
            print("\nCustom modules in loaded model:")
            for name, module in loaded_model.custom_modules.items():
                print(f"  - {name}: {type(module).__name__}")
        else:
            print("\nNo custom modules found in loaded model")
        
        return loaded_model
    except Exception as e:
        print(f"Error during save/load test: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("Creating enhanced YOLOv8 model...")
    model = create_enhanced_yolov8(size='n', pretrained=True)
    
    # Visualize model structure
    visualize_model_structure(model)
    
    # Test detection capabilities
    test_detection(model)
    
    # Test save and load
    loaded_model = test_save_load(model)
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()