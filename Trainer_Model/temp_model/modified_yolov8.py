import os
import torch
import torch.nn as nn
import yaml
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Conv

class GroupNormConv(nn.Module):
    """Conv module with GroupNorm instead of BatchNorm for small feature maps."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        # Use GroupNorm instead of BatchNorm
        # Ensure at least 2 groups but not more than channels/2
        num_groups = min(max(2, c2 // 4), 32)
        self.norm = nn.GroupNorm(num_groups, c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SmallObjectEnhance(nn.Module):
    """Channel attention module optimized for small object detection with GroupNorm."""
    def __init__(self, c1, c2, act=True):
        super().__init__()
        # Use GroupNorm convolutions instead of standard Conv with BatchNorm
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

class GroupNormC2f(nn.Module):
    """C2f module with GroupNorm for better handling of small feature maps."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # Hidden channels
        self.cv1 = GroupNormConv(c1, 2 * self.c, 1, 1)
        self.cv2 = GroupNormConv((2 + n) * self.c, c2, 1)  # Optional GroupNormConv for output
        
        # Create a list of blocks with GroupNorm
        self.m = nn.ModuleList()
        for _ in range(n):
            # Create a bottleneck block with GroupNorm
            block = nn.Sequential(
                GroupNormConv(self.c, self.c, 3),
                GroupNormConv(self.c, self.c, 3)
            )
            self.m.append(block)

    def forward(self, x):
        # Initial convolution
        y = list(self.cv1(x).chunk(2, 1))
        
        # Process through blocks
        for module in self.m:
            y.append(module(y[-1]))
            
        # Concatenate and final convolution
        return self.cv2(torch.cat(y, 1))

class ResidualC2f(nn.Module):
    """C2f block with residual connection and GroupNorm for better gradient flow."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c2f = GroupNormC2f(c1, c2, n, shortcut, g, e)
        self.residual = nn.Identity() if c1 == c2 else GroupNormConv(c1, c2, 1, 1)
        
    def forward(self, x):
        return self.c2f(x) + self.residual(x)

class EnhancedYOLOv8(nn.Module):
    """Enhanced YOLOv8 model using hooks to add custom layers."""
    def __init__(self, size='n', pretrained=True):
        super().__init__()
        
        # Initialize variables to store activations
        self.activations = {}
        self.hooks = []
        
        # Load the base model
        print(f"Loading YOLOv8{size} as base model...")
        self.base_model = YOLO(f"yolov8{size}.pt" if pretrained else f"yolov8{size}.yaml").model
        
        # Store original model attributes
        self.yaml = getattr(self.base_model, 'yaml', None)
        self.names = getattr(self.base_model, 'names', None)
        self.stride = getattr(self.base_model, 'stride', None)
        
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
        """Analyze the YOLOv8 model structure to determine channel sizes."""
        self.channel_sizes = {}
        
        # We'll use a dummy forward pass to analyze the structure
        print("Analyzing base model structure...")
        
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook
        
        # Register temporary hooks to capture layer outputs
        temp_hooks = []
        
        # YOLOv8 model[0] is the backbone
        # model[1] through model[-1] are the detection head components
        for i, module in enumerate(self.base_model.model):
            temp_hooks.append(module.register_forward_hook(get_activation(f"module_{i}")))
        
        # Run a dummy forward pass to get dimensions
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            # Ensure base model is in eval mode during analysis
            self.base_model.eval()
            self.base_model(dummy_input)
        
        # Remove temporary hooks
        for hook in temp_hooks:
            hook.remove()
        
        # Store channel dimensions for layers where we'll insert custom modules
        key_modules = []
        
        # Get all module outputs and note their shapes
        for name, activation in self.activations.items():
            if isinstance(activation, torch.Tensor):
                self.channel_sizes[name] = activation.shape[1]  # Channel dimension
                print(f"{name}: shape {activation.shape}, channels: {activation.shape[1]}")
        
        # Identify key points for insertion (indices can vary by model version)
        # These are typical points after downsampling operations where feature maps change size
        self.insertion_points = []
        
        # Storing all activation names sorted by channel size for easier identification
        sorted_by_channels = sorted(self.channel_sizes.items(), key=lambda x: x[1])
        
        # Get layer names ordered by channel dimension (typically increases after downsampling)
        self.ordered_layers = [name for name, _ in sorted_by_channels]
        
        # Identify key layers based on channel dimensions and position
        # Let's identify layers with unique channel sizes as they typically come after downsampling
        unique_channel_sizes = []
        for _, channels in sorted_by_channels:
            if channels not in unique_channel_sizes:
                unique_channel_sizes.append(channels)
        
        # Get layers with the 2nd, 3rd, and 4th unique channel sizes
        # These typically correspond to layers after downsampling operations
        if len(unique_channel_sizes) >= 4:
            target_channels = unique_channel_sizes[1:4]  # 2nd, 3rd, and 4th unique channel sizes
            
            for name, channels in sorted_by_channels:
                if channels in target_channels and name not in self.insertion_points:
                    self.insertion_points.append(name)
                    target_channels.remove(channels)  # Only get the first layer with this channel size
        
        print(f"Identified insertion points: {self.insertion_points}")
    
    def _create_custom_layers(self):
        """Create custom layers at specific channel sizes."""
        self.custom_modules = nn.ModuleDict()
        
        # Create custom layers for each insertion point
        for i, name in enumerate(self.insertion_points):
            channels = self.channel_sizes[name]
            
            # First insertion point: Add ResidualC2f
            if i == 0:
                self.custom_modules[f"{name}_residual"] = ResidualC2f(channels, channels)
                print(f"Created ResidualC2f layer for {name} with {channels} channels")
            
            # Second insertion point: Add ResidualC2f + SmallObjectEnhance
            elif i == 1:
                self.custom_modules[f"{name}_residual"] = ResidualC2f(channels, channels)
                self.custom_modules[f"{name}_small_obj"] = SmallObjectEnhance(channels, channels)
                print(f"Created ResidualC2f + SmallObjectEnhance layers for {name} with {channels} channels")
            
            # Third insertion point: Add ResidualC2f
            elif i == 2:
                self.custom_modules[f"{name}_residual"] = ResidualC2f(channels, channels)
                print(f"Created ResidualC2f layer for {name} with {channels} channels")
    
    def _register_hooks(self):
        """Register forward hooks to insert custom layer processing."""
        # Clear any existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Define hook functions for each insertion point
        def make_hook(name, custom_module_keys):
            def hook(module, input, output):
                # Process the output through our custom modules
                x = output
                for key in custom_module_keys:
                    # Ensure our custom modules are in the same mode as the model
                    self.custom_modules[key].train(self.training)
                    x = self.custom_modules[key](x)
                return x
            return hook
        
        # Register hooks for each insertion point
        for i, name in enumerate(self.insertion_points):
            idx = int(name.split('_')[1])  # Get module index from name (e.g. "module_3" -> 3)
            
            # Determine which custom modules to apply
            custom_module_keys = []
            if i == 0:  # First insertion: ResidualC2f
                custom_module_keys = [f"{name}_residual"]
            elif i == 1:  # Second insertion: ResidualC2f + SmallObjectEnhance
                custom_module_keys = [f"{name}_residual", f"{name}_small_obj"]
            elif i == 2:  # Third insertion: ResidualC2f
                custom_module_keys = [f"{name}_residual"]
            
            # Register the hook to modify the output
            hook = self.base_model.model[idx].register_forward_hook(make_hook(name, custom_module_keys))
            self.hooks.append(hook)
    
    def forward(self, x):
        """Forward pass - delegates to base model with hooks for modifications."""
        return self.base_model(x)
    
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

def create_enhanced_yolov8(size='n', pretrained=True):
    """
    Create an enhanced YOLOv8 model with custom architecture for small object detection.
    
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
        'insertion_points': model.insertion_points if hasattr(model, 'insertion_points') else None
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
        if 'insertion_points' in data and data['insertion_points'] is not None:
            model.insertion_points = data['insertion_points']
        
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