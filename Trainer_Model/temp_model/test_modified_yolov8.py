import torch
import cv2
import numpy as np
import os
from modified_yolov8 import load_model_with_yaml
import inspect

def test_forward_pass():
    """Test a single forward pass of the enhanced YOLOv8 model and analyze its output."""
    # Load the enhanced YOLOv8 model
    print("Loading Enhanced YOLOv8 model...")
    model_path = "models/enhanced_yolov8n_final.pt"
    model = load_model_with_yaml(model_path, wrap_for_training=False)
    model.eval()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Set your custom classes (for reference)
    custom_classes = {
        0: 'Coin',
        1: 'Drink',
        2: 'Fork',
        3: 'Hammer',
        4: 'Screwdriver',
        5: 'Stapler',
        6: 'sharp-item',
        7: 'toddler'
    }
    model.names = custom_classes
    
    # Create a test image (random RGB tensor)
    test_image = torch.rand(1, 3, 640, 640).to(device)
    print(f"Created test image tensor: {test_image.shape}")
    
    # Display the current 'forward' method implementation for reference
    print("\n" + "="*80)
    print("CURRENT FORWARD METHOD IMPLEMENTATION:")
    print("="*80)
    try:
        forward_method = inspect.getsource(model.forward)
        print(forward_method)
    except Exception as e:
        print(f"Could not get source code for forward method: {e}")
        print("Default method is likely:")
        print("""
    def forward(self, *args, **kwargs):
        # Simply pass all arguments directly to the base model
        return self.base_model(*args, **kwargs)
        """)
    print("="*80 + "\n")
    
    # Test the base model output directly
    print("Testing base_model output directly:")
    with torch.no_grad():
        try:
            base_outputs = model.base_model(test_image)
            print(f"Base model output type: {type(base_outputs)}")
            if isinstance(base_outputs, tuple):
                print(f"Base model tuple length: {len(base_outputs)}")
                print(f"First element type: {type(base_outputs[0])}")
                if isinstance(base_outputs[0], torch.Tensor):
                    print(f"First tensor shape: {base_outputs[0].shape}")
        except Exception as e:
            print(f"Error during base model forward pass: {e}")
    
    print("\nTHIS IS WHERE CONVERSION TO RESULTS SHOULD HAPPEN")
    print("================================================")
    print("The forward method should take the base_outputs (above)")
    print("and convert them to a Results object.\n")
    
    # Test the enhanced model forward method
    print("Testing enhanced model forward method:")
    with torch.no_grad():
        try:
            results = model(test_image)
            print(f"Enhanced model output type: {type(results)}")
            
            if isinstance(results, tuple):
                print(f"Results is a tuple with {len(results)} elements")
                for i, item in enumerate(results):
                    print(f"  Element {i} type: {type(item)}")
                    if isinstance(item, torch.Tensor):
                        print(f"  Element {i} shape: {item.shape}")
            
            # Check for standard Ultralytics Results format
            if hasattr(results, 'boxes'):
                print(f"Results IS a standard Ultralytics Results object with 'boxes' attribute")
                print(f"Number of detections: {len(results.boxes)}")
            else:
                print(f"Results is NOT a standard Ultralytics Results object (no 'boxes' attribute)")
                
            # Check for non-tensor attributes which would indicate a Results object
            if not isinstance(results, (torch.Tensor, tuple)):
                print("Non-tensor attributes:")
                for attr in dir(results):
                    if not attr.startswith('_') and not callable(getattr(results, attr)):
                        print(f"  {attr}: {type(getattr(results, attr))}")
        except Exception as e:
            print(f"Error during enhanced model forward pass: {e}")
            import traceback
            traceback.print_exc()
            results = None
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    if results is None or isinstance(results, (torch.Tensor, tuple)):
        print("The model is not returning a Results object. Fix the forward method to create Results properly.")
        print("\nMake sure you've added this correct implementation to EnhancedYOLOv8.forward():")
        print("""
def forward(self, *args, **kwargs):
    # Get raw outputs
    outputs = self.base_model(*args, **kwargs)
    
    # If outputs are in tuple format, convert to Results
    if isinstance(outputs, tuple):
        try:
            from ultralytics.engine.results import Results
            
            # Extract the outputs from the tuple
            output = outputs[0]  # First element contains detection predictions
            
            # Get original image
            original_img = args[0] if len(args) > 0 else kwargs.get('images', None)
            
            # Create Results object
            if isinstance(original_img, torch.Tensor):
                img = original_img[0] if original_img.shape[0] == 1 else original_img
                img_size = img.shape[1:3] if img.dim() == 3 else img.shape[2:4]  # HW
            else:
                img = original_img
                img_size = None
            
            # Create Results object with the processed predictions
            results = Results(
                boxes=output,  # Detection boxes
                orig_img=img,  # Original image
                names=self.names,  # Class names
                path=None,  # Path to the image
                keypoints=None  # No keypoints for object detection
            )
            
            return results
        except Exception as e:
            print(f"Warning: Could not create Results object: {e}")
            return outputs
    
    # Return outputs directly if already in correct format
    return outputs
        """)
    else:
        print("The model is returning a Results object as expected. If you're still having issues,")
        print("check that your camera test script is correctly handling Results objects.")
    
    print("="*80)
    
    return results

if __name__ == "__main__":
    results = test_forward_pass()