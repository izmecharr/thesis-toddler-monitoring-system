# modified_yolov8.py
import os
import torch
import torch.nn as nn
import yaml
from copy import deepcopy
from datetime import datetime
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Conv, DFL, Detect


class GroupNormConv(nn.Module):
    """Conv module with GroupNorm instead of BatchNorm for small feature maps."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        # Use GroupNorm instead of BatchNorm
        num_groups = min(max(2, c2 // 4), 32)
        self.norm = nn.GroupNorm(num_groups, c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualC2f(nn.Module):
    """C2f block with residual connection for better gradient flow."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c2f = C2f(c1, c2, n, shortcut, g, e)  # Use standard C2f
        self.residual = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        
    def forward(self, x):
        return self.c2f(x) + self.residual(x)


class SmallObjectEnhance(nn.Module):
    """Channel attention module optimized for small object detection with GroupNorm."""
    def __init__(self, c1, c2, act=True):
        super().__init__()
        self.cv1 = GroupNormConv(c1, c2//2, 1, 1, act=act)
        self.cv2 = GroupNormConv(c2//2, c2, 3, 1, act=act)
        
        # Attention mechanism
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            GroupNormConv(c2, c2//16, 1, act=act),
            GroupNormConv(c2//16, c2, 1, act=act),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.cv2(self.cv1(x))
        return x * self.attn(x)


class EnhancedYOLOv8(nn.Module):
    """Enhanced YOLOv8 model using hooks to add custom layers to the backbone."""
    def __init__(self, size='n', pretrained=True):
        super().__init__()
        
        # Initialize variables to store activations
        self.activations = {}
        self.hooks = []
        self.size = size
        
        # Load the base model
        print(f"Loading YOLOv8{size} as base model...")
        self.base_model = YOLO(f"yolov8{size}.pt" if pretrained else f"yolov8{size}.yaml").model
        
        # Store original model attributes
        self.yaml = getattr(self.base_model, 'yaml', None)
        self.names = getattr(self.base_model, 'names', None)
        self.stride = getattr(self.base_model, 'stride', None)
        
        # Fix: Ensure base_model.args has all required attributes
        if hasattr(self.base_model, 'args'):
            self.args = self.base_model.args
            # Make sure hyp is an object with attributes, not a dict
            if isinstance(self.args, dict):
                from types import SimpleNamespace
                self.args = SimpleNamespace(**self.args)
            
            # Ensure required hyperparameters exist
            required_hyps = ['box', 'cls', 'dfl', 'pose', 'kobj', 'label_smoothing', 'nbs']
            for hyp in required_hyps:
                if not hasattr(self.args, hyp):
                    setattr(self.args, hyp, {
                        'box': 7.5,
                        'cls': 0.5,
                        'dfl': 1.5,
                        'pose': 1.0,
                        'kobj': 1.0,
                        'label_smoothing': 0.0,
                        'nbs': 64
                    }.get(hyp, 1.0))
        else:
            # Create default args if not available
            from types import SimpleNamespace
            self.args = SimpleNamespace(
                box=7.5,
                cls=0.5,
                dfl=1.5,
                pose=1.0,
                kobj=1.0,
                label_smoothing=0.0,
                nbs=64
            )
        
        # Ensure base_model also has proper args
        self.base_model.args = self.args
        
        # Analyze model structure to determine where to place custom modules
        self._analyze_model_structure()
        
        # Create custom layers based on analysis
        self._create_custom_layers()
        
        # Register hooks to capture activations at specific points
        self._register_hooks()
        
        # Print model info
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Enhanced YOLOv8{size} model created with {num_params:,} parameters")

    def _analyze_model_structure(self):
        """Analyze the YOLOv8 model structure to find C2f blocks."""
        self.channel_sizes = {}
        self.c2f_indices = []
        
        print("\nAnalyzing base model structure...")
        
        # Find C2f blocks in the backbone
        for i, module in enumerate(self.base_model.model):
            if isinstance(module, C2f):
                self.c2f_indices.append(i)
                
        # Get channel sizes
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook
        
        # Register temporary hooks to capture layer outputs
        temp_hooks = []
        for i, module in enumerate(self.base_model.model):
            temp_hooks.append(module.register_forward_hook(get_activation(f"module_{i}")))
        
        # Run a dummy forward pass to get dimensions
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            self.base_model.eval()
            self.base_model(dummy_input)
        
        # Remove temporary hooks
        for hook in temp_hooks:
            hook.remove()
        
        # Store channel dimensions for C2f layers
        for idx in self.c2f_indices:
            if f"module_{idx}" in self.activations:
                output = self.activations[f"module_{idx}"]
                if isinstance(output, torch.Tensor):
                    self.channel_sizes[idx] = output.shape[1]
                    print(f"C2f layer {idx}: shape {output.shape}, channels: {output.shape[1]}")
        
        print(f"\nFound C2f blocks at indices: {self.c2f_indices}")
    
    def _create_custom_layers(self):
        """Create custom layers at specific C2f positions."""
        self.custom_modules = nn.ModuleDict()
        
        # Get the first 3 C2f blocks from the backbone (ignore head)
        backbone_c2f = self.c2f_indices[:3]
        
        # Create custom layers for each C2f position
        if len(backbone_c2f) >= 3:
            # After first C2f (layer 2): ResidualC2f
            idx1 = backbone_c2f[0]
            c1 = self.channel_sizes[idx1]
            self.custom_modules[f"layer_{idx1}_residual"] = ResidualC2f(c1, c1)
            print(f"Created ResidualC2f after layer {idx1} with {c1} channels")
            
            # After second C2f (layer 4): SmallObjectEnhance
            idx2 = backbone_c2f[1]
            c2 = self.channel_sizes[idx2]
            self.custom_modules[f"layer_{idx2}_small_obj"] = SmallObjectEnhance(c2, c2)
            print(f"Created SmallObjectEnhance after layer {idx2} with {c2} channels")
            
            # After third C2f (layer 6): ResidualC2f
            idx3 = backbone_c2f[2]
            c3 = self.channel_sizes[idx3]
            self.custom_modules[f"layer_{idx3}_residual"] = ResidualC2f(c3, c3)
            print(f"Created ResidualC2f after layer {idx3} with {c3} channels")
    
    def _register_hooks(self):
        """Register forward hooks to insert custom layer processing."""
        # Clear any existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Get backbone C2f indices
        backbone_c2f = self.c2f_indices[:3]
        
        # Define hook functions
        if len(backbone_c2f) >= 3:
            # Hook for first C2f: ResidualC2f
            def hook1(module, input, output):
                idx = backbone_c2f[0]
                custom_key = f"layer_{idx}_residual"
                self.custom_modules[custom_key].train(self.training)
                return self.custom_modules[custom_key](output)
            
            # Hook for second C2f: SmallObjectEnhance
            def hook2(module, input, output):
                idx = backbone_c2f[1]
                custom_key = f"layer_{idx}_small_obj"
                self.custom_modules[custom_key].train(self.training)
                return self.custom_modules[custom_key](output)
            
            # Hook for third C2f: ResidualC2f
            def hook3(module, input, output):
                idx = backbone_c2f[2]
                custom_key = f"layer_{idx}_residual"
                self.custom_modules[custom_key].train(self.training)
                return self.custom_modules[custom_key](output)
            
            # Register hooks
            self.hooks.append(self.base_model.model[backbone_c2f[0]].register_forward_hook(hook1))
            self.hooks.append(self.base_model.model[backbone_c2f[1]].register_forward_hook(hook2))
            self.hooks.append(self.base_model.model[backbone_c2f[2]].register_forward_hook(hook3))
    
    def forward(self, *args, **kwargs):
        """Forward pass that returns Results objects."""
        # Get raw outputs from base model (with hooks for our enhanced layers)
        outputs = self.base_model(*args, **kwargs)
        
        # If outputs are in tuple format, convert to Results
        if isinstance(outputs, tuple):
            try:
                from ultralytics.engine.results import Results
                from ultralytics.utils.ops import non_max_suppression
                import inspect
                import torch
                
                # Get original image
                original_img = args[0] if len(args) > 0 else kwargs.get('images', None)
                
                # CRITICAL: Process through NMS first
                # First element of outputs contains predictions
                preds = outputs[0]
                
                # Apply non-maximum suppression
                # If preds has shape [batch, anchors, classes+5]
                if len(preds.shape) == 3:
                    # Process through NMS to get final detections
                    # Default parameters: conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False
                    nms_preds = non_max_suppression(preds, 0.25, 0.45)
                    
                    # non_max_suppression returns a list of tensors, but Results expects a tensor
                    # For a single image, we can take the first element
                    if len(nms_preds) > 0 and len(nms_preds[0]) > 0:
                        # Convert to tensor if it's a list
                        if isinstance(nms_preds[0], list):
                            boxes_tensor = torch.tensor(nms_preds[0])
                        else:
                            # It's already a tensor
                            boxes_tensor = nms_preds[0]
                    else:
                        # No detections, create an empty tensor with the right shape
                        boxes_tensor = torch.zeros((0, 6), device=preds.device)
                else:
                    # Already in the right format
                    boxes_tensor = preds
                
                # Create Results object
                if isinstance(original_img, torch.Tensor):
                    img = original_img[0] if original_img.shape[0] == 1 else original_img
                else:
                    img = original_img
                    
                # Create a parameters dictionary with only compatible parameters
                results_params = {
                    'orig_img': img,       # Original image
                    'names': self.names,   # Class names
                    'boxes': boxes_tensor, # Detection boxes (now as a tensor)
                }
                
                # Add path parameter
                results_params['path'] = None
                    
                # Create Results object with compatible parameters
                results = Results(**results_params)
                
                return results
            except Exception as e:
                import traceback
                print(f"Warning: Error creating Results object: {e}")
                traceback.print_exc()
                
                # If all else fails, try to use the model's predict method directly
                if hasattr(self.base_model, 'predict'):
                    try:
                        print("Falling back to base_model.predict()...")
                        return self.base_model.predict(*args, **kwargs)
                    except Exception as pred_error:
                        print(f"Error with base_model.predict(): {pred_error}")
                
                return outputs
        
        # Return outputs directly if already in correct format
        return outputs
    
    def eval(self):
        """Switch to evaluation mode."""
        super().eval()
        self.base_model.eval()
        for module in self.custom_modules.values():
            module.eval()
        return self
    
    def train(self, mode=True):
        """Switch to training mode."""
        super().train(mode)
        if mode:
            self.base_model.train()
            for module in self.custom_modules.values():
                module.train()
        else:
            self.base_model.eval()
            for module in self.custom_modules.values():
                module.eval()
        return self
    
    @property
    def nc(self):
        """Number of classes."""
        return getattr(self.base_model, 'nc', None)
    
    @nc.setter
    def nc(self, value):
        """Set number of classes."""
        if hasattr(self.base_model, 'nc'):
            self.base_model.nc = value
            
    def predict(self, *args, **kwargs):
        """Prediction method that delegates to base model."""
        if hasattr(self.base_model, 'predict'):
            return self.base_model.predict(*args, **kwargs)
        raise NotImplementedError("Base model does not have a predict method")
    
    def val(self, *args, **kwargs):
        """Validation method that delegates to base model."""
        if hasattr(self.base_model, 'val'):
            return self.base_model.val(*args, **kwargs)
        raise NotImplementedError("Base model does not have a val method")


def create_enhanced_yolov8(size='n', pretrained=True):
    """
    Create an enhanced YOLOv8 model with custom backbone architecture for small object detection.
    
    Args:
        size (str): Model size 'n', 's', 'm', 'l', or 'x'
        pretrained (bool): Whether to load pretrained weights
    
    Returns:
        Enhanced YOLOv8 model
    """
    model = EnhancedYOLOv8(size=size, pretrained=pretrained)
    # Set to eval mode by default for inference
    model.eval()
    return model


def save_model_with_yaml(model, path):
    """Save model with its yaml configuration."""
    # Create a dictionary to store the model state and configuration
    save_dict = {
        'model': model.state_dict(),
        'base_model': model.base_model.state_dict() if hasattr(model, 'base_model') else None,
        'yaml': model.yaml if hasattr(model, 'yaml') else None,
        'names': model.names if hasattr(model, 'names') else None,
        'stride': model.stride if hasattr(model, 'stride') else None,
        'custom_modules': {name: module.state_dict() for name, module in 
                           model.custom_modules.items()} if hasattr(model, 'custom_modules') else None,
        'channel_sizes': model.channel_sizes if hasattr(model, 'channel_sizes') else None,
        'c2f_indices': model.c2f_indices if hasattr(model, 'c2f_indices') else None
    }
    
    # Save the dictionary
    torch.save(save_dict, path)
    print(f"Model saved to {path}")


def load_model_with_yaml(path):
    """Load enhanced model from saved state."""
    # Load the saved state
    data = torch.load(path, map_location='cpu')
    
    # Determine model size from parameter count or filename
    size = 'n'  # Default to nano
    if 'yolov8s' in path.lower():
        size = 's'
    elif 'yolov8m' in path.lower():
        size = 'm'
    elif 'yolov8l' in path.lower():
        size = 'l'
    elif 'yolov8x' in path.lower():
        size = 'x'
    
    # Create a new model instance
    model = create_enhanced_yolov8(size=size, pretrained=False)
    
    # Load state dictionary if it's our enhanced model format
    if 'custom_modules' in data and data['custom_modules'] is not None:
        print("Loading enhanced model state...")
        
        # Load base model state
        if 'base_model' in data and data['base_model'] is not None:
            model.base_model.load_state_dict(data['base_model'])
        
        # Load custom modules
        for name, state in data['custom_modules'].items():
            if name in model.custom_modules:
                model.custom_modules[name].load_state_dict(state)
        
        # Restore configuration data
        if 'names' in data and data['names'] is not None:
            model.names = data['names']
        if 'stride' in data and data['stride'] is not None:
            model.stride = data['stride']
        if 'yaml' in data and data['yaml'] is not None:
            model.yaml = data['yaml']
        
        # Restore structural data
        if 'channel_sizes' in data and data['channel_sizes'] is not None:
            model.channel_sizes = data['channel_sizes']
        if 'c2f_indices' in data and data['c2f_indices'] is not None:
            model.c2f_indices = data['c2f_indices']
        
        print("Enhanced model loaded successfully")
    else:
        # If it's a standard YOLO model, create a new enhanced model from it
        print("Loading as standard YOLO model and enhancing it...")
        standard_model = YOLO(path)
        enhanced_model = create_enhanced_yolov8(size=size, pretrained=False)
        
        # Transfer weights from standard model base to enhanced model base
        enhanced_model.base_model.load_state_dict(standard_model.model.state_dict())
        model = enhanced_model
    
    # Ensure model is in eval mode after loading
    model.eval()
    return model