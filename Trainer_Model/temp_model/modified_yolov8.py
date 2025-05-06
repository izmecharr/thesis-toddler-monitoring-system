# modified_yolov8.py
import os
import torch
import torch.nn as nn
import yaml
from copy import deepcopy
from datetime import datetime
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Conv, DFL, Detect
from ultralytics.models.yolo.detect.train import DetectionTrainer

try:
    from ultralytics.engine.model import ModelEMA
except ImportError:
    # Create a fallback EMA implementation if not available
    class ModelEMA:
        def __init__(self, model, decay=0.9999):
            self.ema = model
            self.decay = decay
            self.updates = 0
            
        def update(self, model):
            self.updates += 1
            
        def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
            pass

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
        self.task = getattr(self.base_model, 'task', None)
        
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
        
        # Extract the Detect module for later reference
        self.detect_module = None
        for module in self.base_model.modules():
            if isinstance(module, Detect):
                self.detect_module = module
                break
                
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
                print(f"Found C2f block at index {i}")
        
        # Get channel sizes from model structure directly without forward pass
        # This is a more robust approach than running a dummy forward pass
        prev_channels = 3  # Starting with RGB input
        estimated_sizes = {}
        
        for i, module in enumerate(self.base_model.model):
            # For C2f modules, we can estimate the output channels
            if isinstance(module, C2f):
                # In YOLOv8, C2f typically keeps the same number of channels 
                # or uses the c2 parameter if defined
                if hasattr(module, 'c2'):
                    estimated_sizes[i] = module.c2
                elif hasattr(module, 'cv2') and hasattr(module.cv2, 'conv') and hasattr(module.cv2.conv, 'out_channels'):
                    estimated_sizes[i] = module.cv2.conv.out_channels
                else:
                    # Fallback to previous channels if we can't determine
                    estimated_sizes[i] = prev_channels
                
                prev_channels = estimated_sizes[i]
                print(f"Estimated C2f layer {i} output channels: {estimated_sizes[i]}")
        
        # Store the estimated channel sizes
        self.channel_sizes = estimated_sizes
        
        print(f"\nFound C2f blocks at indices: {self.c2f_indices}")
        
        # Set default channel sizes if we couldn't determine them
        # These are based on typical YOLOv8n architecture
        if not self.channel_sizes:
            typical_channels = {
                2: 64,   # First C2f block typically has 64 channels in YOLOv8n
                4: 128,  # Second C2f block typically has 128 channels
                6: 256   # Third C2f block typically has 256 channels
            }
            for i, channels in typical_channels.items():
                if i in self.c2f_indices and i not in self.channel_sizes:
                    self.channel_sizes[i] = channels
                    print(f"Using default channel size for C2f layer {i}: {channels}")
    
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
    
    # NEW: Add compatibility methods for the training pipeline
    def get_model(self):
        """Return base model for compatibility."""
        return self.base_model
        
    def model(self, *args, **kwargs):
        """Handle model calls from training pipeline."""
        return self.forward(*args, **kwargs)
    
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
    
# Add a new custom loader function to replace attempt_load_one_weight
def attempt_load_enhanced_model(model, device=''):
    """Custom model loader for our enhanced YOLOv8 model."""
    from pathlib import Path
    import torch
    
    if isinstance(model, (str, Path)):
        # Try to load as our enhanced model
        try:
            enhanced_model = load_model_with_yaml(model)
            print(f"Successfully loaded as enhanced model: {model}")
            enhanced_model.to(device)
            return enhanced_model, {}
        except Exception as e:
            print(f"Failed to load as enhanced model: {e}")
            # Let standard YOLO loading handle it
            return None, None
    else:
        # Already a model object
        model.to(device)
        return model, {}

class EnhancedDetectionTrainer:
    """Patch class for training our enhanced model."""
    
    @staticmethod
    def patch_trainer(trainer):
        """Patch a standard DetectionTrainer instance to work with our enhanced model."""
        # Store original method
        original_setup_model = trainer.setup_model
        
        # Create a patched setup_model method
        def patched_setup_model():
            """Custom setup_model implementation."""
            # Try to load as enhanced model first
            model, ckpt = attempt_load_enhanced_model(trainer.model, device=trainer.device)
            
            if model is not None:
                # Successfully loaded as enhanced model
                print("Using enhanced model for training")
                trainer.model = model
                
                # Initialize AMP if applicable
                trainer.amp = torch.cuda.is_available() and getattr(trainer.args, 'amp', False)
                trainer.scaler = torch.cuda.amp.GradScaler(enabled=trainer.amp)
                
                # Create EMA model if validation is enabled
                if not getattr(trainer.args, 'noval', False):
                    trainer.ema = ModelEMA(model)
                
                # Ensure model is in training mode
                model.train()
                
                # Return the model and empty checkpoint
                return model, {}
            else:
                # Fall back to original method if not an enhanced model
                print("Enhanced model loading failed, falling back to standard loader")
                return original_setup_model()
        
        # Replace the method
        trainer.setup_model = patched_setup_model
        
        # Store original save_model method
        original_save_model = trainer.save_model
        
        # Create a patched save_model method
        def patched_save_model(file=''):
            """Custom save_model implementation for enhanced models."""
            if file == '':
                file = trainer.best_model_path if trainer.best_fitness == trainer.fitness else trainer.last_model_path
            
            if hasattr(trainer.model, 'is_enhanced') or hasattr(trainer.model, 'custom_modules'):
                # Save enhanced model
                save_model_with_yaml(trainer.model, str(file))
                print(f"Saved enhanced model to {file}")
            else:
                # Use original method for standard models
                original_save_model(file)
            
            return str(file)
        
        # Replace the method
        trainer.save_model = patched_save_model
        
        # Patch model validation to work with enhanced models
        original_validate = trainer.validate
        
        def patched_validate():
            """Patched validate to work with enhanced models."""
            # Using a try/except to ensure we fall back gracefully
            try:
                return original_validate()
            except Exception as e:
                print(f"Warning: Standard validation failed: {e}")
                print("Using alternative validation approach")
                
                # Simple validation
                if hasattr(trainer.model, 'val'):
                    return trainer.model.val(data=trainer.args.data)
                elif hasattr(trainer.validator, 'model'):
                    trainer.validator.model = trainer.model
                    return trainer.validator.validate()
                else:
                    print("Warning: Validation not performed")
                    return {}
        
        # Replace the method
        trainer.validate = patched_validate
        
        return trainer

class EnhancedModelWrapper(torch.nn.Module):
    """Wrapper for our enhanced model that makes it compatible with YOLOv8 training pipeline."""
    
    def __init__(self, enhanced_model):
        super().__init__()
        self.model = enhanced_model
        
        # Copy attributes from the enhanced model
        for attr_name in ['nc', 'names', 'stride', 'yaml', 'args', 'task']:
            if hasattr(enhanced_model, attr_name):
                setattr(self, attr_name, getattr(enhanced_model, attr_name))
        
        # Add special flag to identify as an enhanced model
        self.is_enhanced_wrapper = True
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to the enhanced model."""
        return self.model(*args, **kwargs)
    
    def to(self, *args, **kwargs):
        """Move model to device."""
        self.model.to(*args, **kwargs)
        return self
    
    def train(self, mode=True):
        """Set training mode."""
        self.model.train(mode)
        super().train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.model.eval()
        super().eval()
        return self
    
    def state_dict(self, *args, **kwargs):
        """Get state dictionary for saving."""
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dictionary."""
        return self.model.load_state_dict(state_dict, strict)
    
    def modules(self, *args, **kwargs):
        """Get model modules."""
        return self.model.modules(*args, **kwargs)
    
    def named_modules(self, *args, **kwargs):
        """Get named modules."""
        return self.model.named_modules(*args, **kwargs)
    
    def parameters(self, *args, **kwargs):
        """Get model parameters."""
        return self.model.parameters(*args, **kwargs)
    
    def named_parameters(self, *args, **kwargs):
        """Get named parameters."""
        return self.model.named_parameters(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to the enhanced model."""
        if name in ['model', 'is_enhanced_wrapper'] or name.startswith('__'):
            return super().__getattr__(name)
        return getattr(self.model, name)

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
    
    # NEW: Add a flag to identify this as an enhanced model
    setattr(model, 'is_enhanced', True)
    
    # Set to eval mode by default for inference
    model.eval()
    return model


def save_model_with_yaml(model, path):
    """Save model with its yaml configuration."""
    # Unwrap if it's a wrapper
    if hasattr(model, 'is_enhanced_wrapper') and model.is_enhanced_wrapper:
        model = model.model
    
    # Create a dictionary to store the model state and configuration
    save_dict = {
        'model': model.state_dict() if not hasattr(model, 'is_enhanced_wrapper') else model.model.state_dict(),
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

def load_model_with_yaml(path, wrap_for_training=True):
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
        try:
            standard_model = YOLO(path)
            enhanced_model = create_enhanced_yolov8(size=size, pretrained=False)
            
            # Transfer weights from standard model base to enhanced model base
            enhanced_model.base_model.load_state_dict(standard_model.model.state_dict())
            model = enhanced_model
        except Exception as e:
            print(f"Error loading as standard model: {e}")
            print("Using pretrained weights instead")
            model = create_enhanced_yolov8(size=size, pretrained=True)
    
    # Ensure model is in eval mode after loading
    model.eval()
    
    # Wrap the model if requested (for training compatibility)
    if wrap_for_training:
        return EnhancedModelWrapper(model)
    else:
        return model