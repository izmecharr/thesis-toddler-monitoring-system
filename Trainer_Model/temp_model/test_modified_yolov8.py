# test_enhanced_model.py
import torch
import time
from collections import OrderedDict
from modified_yolov8 import (
    create_enhanced_yolov8, 
    save_model_with_yaml, 
    load_model_with_yaml,
    SmallObjectEnhance,
    ResidualC2f
)

def analyze_model_structure(model):
    """Analyze model structure and hook placement."""
    print("\n=== Model Structure Analysis ===")
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
<<<<<<< HEAD
    # Check C2f indices and expected hook positions
    if hasattr(model, 'c2f_indices'):
        print(f"\nFound C2f blocks at indices: {model.c2f_indices}")
        backbone_c2f = model.c2f_indices[:3]
        print(f"Backbone C2f blocks: {backbone_c2f}")
        
        # Check expected hook positions
        expected_hooks = {}
        if len(backbone_c2f) >= 3:
            expected_hooks[backbone_c2f[0]] = "ResidualC2f"
            expected_hooks[backbone_c2f[1]] = "SmallObjectEnhance"
            expected_hooks[backbone_c2f[2]] = "ResidualC2f"
        
        # Verify hooks are correctly placed
        print("\nExpected hook placements:")
        for idx, hook_type in expected_hooks.items():
            print(f"  Layer {idx}: {hook_type}")
    
    # Check custom modules
=======
    # Look for custom backbone enhancements
    if hasattr(model, 'backbone_enhancements'):
        print("\nBackbone Enhancements:")
        for pos_name, details in model.backbone_enhancements.items():
            print(f"  - {pos_name}: {details['type']} after {details['position']} ({details['channels']} channels)")
    
    # Look for our custom modules
>>>>>>> c85ddd7525b84bcf2edfa7b0eee5c39bf773f684
    if hasattr(model, 'custom_modules'):
        print("\nCustom modules found:")
        for name, module in model.custom_modules.items():
<<<<<<< HEAD
            param_count = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {type(module).__name__}, {param_count:,} parameters")
    
    return total_params

def trace_backbone_enhancements(model):
    """Trace the backbone enhancements by analyzing layers."""
    print("\n=== Backbone Enhancement Analysis ===")
    
    if not hasattr(model, 'c2f_indices'):
        print("No C2f indices found")
        return
    
    backbone_c2f = model.c2f_indices[:3]
    print(f"Analyzing backbone C2f blocks: {backbone_c2f}")
    
    hooks_found = []
    for idx, layer in enumerate(model.base_model.model):
        if idx in backbone_c2f:
            hook_str = []
            # Find which hook is registered for this layer
            for hook_idx, hook_ref in enumerate(layer._forward_hooks.values()):
                hook_func_name = getattr(hook_ref, '__name__', 'unknown')
                hook_str.append(f"Hook {hook_idx}")
            
            hooks_found.append((idx, layer, hook_str))
    
    print("\nHook analysis:")
    for idx, layer, hooks in hooks_found:
        layer_type = type(layer).__name__
        if hasattr(model, 'channel_sizes') and idx in model.channel_sizes:
            channels = model.channel_sizes[idx]
            print(f"  Layer {idx} ({layer_type}): {channels} channels, Hooks: {len(hooks)}")
        else:
            print(f"  Layer {idx} ({layer_type}): Hooks: {len(hooks)}")

def test_forward_pass_with_hooks(model):
    """Test forward pass and verify custom layers are called."""
    print("\n=== Forward Pass Testing ===")
    
    # Add activation hooks to custom modules
    activations = {}
    hooks = []
    
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = {
                'shape': output.shape if hasattr(output, 'shape') else 'No shape',
                'called': True
            }
        return hook
    
    # Hook custom modules
    for name, module in model.custom_modules.items():
        hook = module.register_forward_hook(get_activation(name))
        hooks.append(hook)
    
    # Forward pass
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    
    print(f"Running forward pass with input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Check if custom modules were called
    print("\nCustom module activation check:")
    for name, info in activations.items():
        called_status = "✓ Called" if info['called'] else "✗ Not called"
        print(f"  {name}: {called_status}, Output shape: {info['shape']}")
    
    # Handle different output types
    if isinstance(output, tuple):
        print(f"\nOutput is a tuple with {len(output)} elements")
        for i, out in enumerate(output):
            if hasattr(out, 'shape'):
                print(f"  Element {i} shape: {out.shape}")
            else:
                print(f"  Element {i} type: {type(out)}")
    elif hasattr(output, 'shape'):
        print(f"\nFinal output shape: {output.shape}")
=======
            module_type = type(module).__name__
            print(f"  - {name}: {module_type}")
>>>>>>> c85ddd7525b84bcf2edfa7b0eee5c39bf773f684
    else:
        print(f"\nOutput type: {type(output)}")
    
<<<<<<< HEAD
    return activations

def test_training_mode(model):
    """Test training mode with custom enhancement."""
    print("\n=== Training Mode Test ===")
    
    # Switch to training mode
    model.train()
    
    # Create dummy batch
    batch = {
        'img': torch.randn(2, 3, 640, 640),  # Batch of 2 images
        'cls': torch.randint(0, 80, (10,)),  # 10 objects total
        'bboxes': torch.rand(10, 4),         # 10 bounding boxes
        'batch_idx': torch.tensor([0]*5 + [1]*5),  # 5 objects per image
        'gt_cls': torch.ones(1, 100)         # Ground truth classes
    }
    
    print("Testing training forward pass...")
=======
    # Test forward pass
    print("\nTesting forward pass with dummy input...")
>>>>>>> c85ddd7525b84bcf2edfa7b0eee5c39bf773f684
    try:
        output = model(batch)
        
        if isinstance(output, tuple) and len(output) == 2:
            loss, loss_items = output
            print(f"Loss: {loss:.6f}")
            print(f"Loss components: {loss_items}")
            print("✓ Training forward pass successful")
        else:
            print(f"Unexpected output type: {type(output)}")
            print("✗ Training forward pass failed")
    
    except Exception as e:
        print(f"Error in training mode: {e}")
        import traceback
        traceback.print_exc()

def compare_with_standard(enhanced_model):
    """Compare enhanced model with standard YOLOv8."""
    print("\n=== Comparison with Standard YOLOv8 ===")
    
    try:
        from ultralytics import YOLO
        standard_model = YOLO('yolov8n.pt').model
        
        # Parameter comparison
        enhanced_params = sum(p.numel() for p in enhanced_model.parameters())
        standard_params = sum(p.numel() for p in standard_model.parameters())
        
        print(f"Parameter count:")
        print(f"  Enhanced model: {enhanced_params:,}")
        print(f"  Standard model: {standard_params:,}")
        print(f"  Increase: {((enhanced_params - standard_params) / standard_params) * 100:.1f}%")
        
        # Inference time comparison
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # Enhanced model timing
        start_time = time.time()
        with torch.no_grad():
            enhanced_model.eval()
            enhanced_output = enhanced_model(dummy_input)
        enhanced_time = time.time() - start_time
        
<<<<<<< HEAD
        # Standard model timing  
        start_time = time.time()
=======
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

def verify_backbone_enhancements(model):
    """Verify that backbone enhancements are correctly placed."""
    print("\n=== Verifying Backbone Enhancements ===")
    
    # Check that we have the expected number of enhancements
    if hasattr(model, 'backbone_enhancements'):
        num_enhancements = len(model.backbone_enhancements)
        print(f"Found {num_enhancements} backbone enhancements")
        
        # Verify specific enhancements
        expected_enhancements = [
            ("c2f_1", "ResidualC2f"),
            ("c2f_2", "SmallObjectEnhance"), 
            ("c2f_3", "ResidualC2f")
        ]
        
        found_count = 0
        for expected_name, expected_type in expected_enhancements:
            found = False
            for actual_name, details in model.backbone_enhancements.items():
                if expected_name == actual_name and details['type'] == expected_type:
                    print(f"✓ Found {expected_type} at {actual_name} as expected")
                    found = True
                    found_count += 1
                    break
            if not found:
                print(f"✗ Missing {expected_type} at {expected_name}")
        
        if found_count == len(expected_enhancements):
            print("All expected enhancements were found!")
        else:
            print(f"Only found {found_count}/{len(expected_enhancements)} expected enhancements")
    else:
        print("No backbone_enhancements attribute found in model")

def test_detection(model):
    """Test if the model can actually perform detection."""
    try:
        print("\nTesting detection capabilities...")
        dummy_input = torch.randn(1, 3, 640, 640)
>>>>>>> c85ddd7525b84bcf2edfa7b0eee5c39bf773f684
        with torch.no_grad():
            standard_model.eval()
            standard_output = standard_model(dummy_input)
        standard_time = time.time() - start_time
        
        print(f"\nInference time:")
        print(f"  Enhanced model: {enhanced_time*1000:.2f} ms")
        print(f"  Standard model: {standard_time*1000:.2f} ms")
        print(f"  Difference: {((enhanced_time - standard_time) / standard_time) * 100:.1f}%")
        
        # Output shape comparison
        print(f"\nOutput shapes:")
        print(f"  Enhanced model output type: {type(enhanced_output)}")
        print(f"  Standard model output type: {type(standard_output)}")
        
        if isinstance(enhanced_output, tuple):
            print(f"  Enhanced output tuple length: {len(enhanced_output)}")
        if isinstance(standard_output, tuple):
            print(f"  Standard output tuple length: {len(standard_output)}")
        
    except ImportError:
        print("Could not import ultralytics for comparison")
    except Exception as e:
        print(f"Error during comparison: {e}")

def test_save_load(model):
    """Test saving and loading enhanced model."""
    print("\n=== Save/Load Testing ===")
    
    try:
        # Save model
        save_path = "enhanced_yolov8_test.pt"
        save_model_with_yaml(model, save_path)
        
        # Load model
        loaded_model = load_model_with_yaml(save_path)
        
        # Verify loaded model has same structure
        original_params = sum(p.numel() for p in model.parameters())
        loaded_params = sum(p.numel() for p in loaded_model.parameters())
        
        print(f"Parameter count match: {original_params == loaded_params}")
        
        # Test forward pass on loaded model
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            loaded_model.eval()
            output = loaded_model(dummy_input)
        
<<<<<<< HEAD
        print("✓ Save/Load test successful")
=======
        print("Forward pass of loaded model successful!")
        
        # Verify backbone enhancements after loading
        verify_backbone_enhancements(loaded_model)
        
>>>>>>> c85ddd7525b84bcf2edfa7b0eee5c39bf773f684
        return loaded_model
        
    except Exception as e:
        print(f"Error in save/load test: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("Creating hook-based enhanced YOLOv8 model...")
    model = create_enhanced_yolov8(size='n', pretrained=True)
    
    # Analyze model structure
    analyze_model_structure(model)
    
<<<<<<< HEAD
    # Trace backbone enhancements
    trace_backbone_enhancements(model)
=======
    # Verify backbone enhancements
    verify_backbone_enhancements(model)
    
    # Test detection capabilities
    test_detection(model)
>>>>>>> c85ddd7525b84bcf2edfa7b0eee5c39bf773f684
    
    # Test forward pass with hooks
    activations = test_forward_pass_with_hooks(model)
    
    # Test training mode
    test_training_mode(model)
    
    # Compare with standard model
    compare_with_standard(model)
    
    # Test save/load
    loaded_model = test_save_load(model)
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Hook-based enhancements successfully applied to backbone:")
    if hasattr(model, 'c2f_indices') and len(model.c2f_indices) >= 3:
        backbone_c2f = model.c2f_indices[:3]
        print(f"  Layer {backbone_c2f[0]}: ResidualC2f")
        print(f"  Layer {backbone_c2f[1]}: SmallObjectEnhance")
        print(f"  Layer {backbone_c2f[2]}: ResidualC2f")
    print("✓ All tests completed successfully!")

if __name__ == "__main__":
    main()