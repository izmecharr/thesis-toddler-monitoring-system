import torch
import yaml
import os
import glob
import numpy as np
import cv2

class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, yaml_path, split='train', transforms=None):
        """
        Dataset for YOLO format data based on a yaml config file
        
        Args:
            yaml_path: Path to data.yaml file
            split: 'train', 'val', or 'test'
            transforms: Image transformations
        """
        self.transforms = transforms
        self.split = split
        
        # Load dataset configuration
        with open(yaml_path, 'r') as f:
            self.data_cfg = yaml.safe_load(f)
        
        # Verify the required keys exist in the YAML
        required_keys = ['train', 'val' if split != 'test' else split, 'nc', 'names']
        for key in required_keys:
            if key not in self.data_cfg:
                raise KeyError(f"data.yaml must contain '{key}' key")
        
        # Get image directory from YAML
        if split == 'train':
            self.img_dir = self.data_cfg['train']
        elif split == 'val':
            self.img_dir = self.data_cfg['val']
        elif split == 'test':
            if 'test' not in self.data_cfg:
                print(f"Warning: 'test' not found in data.yaml, using 'val' instead")
                self.img_dir = self.data_cfg['val']
            else:
                self.img_dir = self.data_cfg['test']
        else:
            raise ValueError(f"Invalid split: {split}")
            
        # Infer label directory - usually replaces 'images' with 'labels' in path
        if 'labels' in self.data_cfg and split in self.data_cfg['labels']:
            # Use explicit label path if provided
            self.label_dir = self.data_cfg['labels'][split]
        else:
            # Otherwise infer from image path
            self.label_dir = self.img_dir.replace('images', 'labels')
        
        # Get class names and count from YAML
        self.class_names = self.data_cfg['names']
        self.num_classes = self.data_cfg['nc']
        
        # Ensure number of classes matches names list
        assert len(self.class_names) == self.num_classes, \
            f"Number of classes ({self.num_classes}) doesn't match names list length ({len(self.class_names)})"
        
        # Get image paths
        self.img_paths = self._get_img_paths()
        
        print(f"Loaded {split} dataset from '{self.img_dir}'")
        print(f"Found {len(self.img_paths)} images with {self.num_classes} classes")
        print(f"Label directory: '{self.label_dir}'")
        
    def _get_img_paths(self):
        """Get all image paths in the directory"""
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        img_paths = []
        
        for ext in img_extensions:
            img_paths.extend(glob.glob(os.path.join(self.img_dir, f'**/*{ext}'), recursive=True))
        
        return sorted(img_paths)
    
    def _get_label_path(self, img_path):
        """Convert image path to corresponding label path"""
        # Get relative path of image in image directory
        img_rel_path = os.path.relpath(img_path, self.img_dir)
        # Replace file extension with .txt
        base_name = os.path.splitext(img_rel_path)[0]
        # Construct label path
        label_path = os.path.join(self.label_dir, f"{base_name}.txt")
        return label_path
    
    def _load_labels(self, label_path):
        """Load YOLO format labels from .txt file with robust error handling"""
        if not os.path.exists(label_path):
            return torch.zeros((0, 5))  # Return empty tensor if no labels
        
        try:
            # First try with np.loadtxt, which is fast but strict
            labels = np.loadtxt(label_path, delimiter=' ', ndmin=2).astype(np.float32)
            
            # Ensure we only get the first 5 columns (class, x, y, w, h) if there are more
            if labels.shape[1] > 5:
                labels = labels[:, :5]
                
            return torch.from_numpy(labels)
        except Exception as e:
            # If loadtxt fails, use a more robust approach with manual parsing
            print(f"Error with standard loading for {label_path}: {str(e)}")
            print(f"Attempting fallback loading method...")
            
            try:
                # Try manual line-by-line parsing
                labels = []
                with open(label_path, 'r') as f:
                    for line in f:
                        try:
                            # Split the line and take only the first 5 values
                            parts = line.strip().split()
                            if len(parts) >= 5:  # Must have at least class, x, y, w, h
                                # Convert the first 5 elements to float
                                label = [float(p) for p in parts[:5]]
                                labels.append(label)
                        except Exception as line_e:
                            print(f"Skipping malformed line in {label_path}: {line.strip()}")
                
                if labels:
                    return torch.tensor(labels, dtype=torch.float32)
                else:
                    print(f"No valid lines found in {label_path}")
                    return torch.zeros((0, 5))
                    
            except Exception as fallback_e:
                print(f"Fallback loading failed for {label_path}: {str(fallback_e)}")
                return torch.zeros((0, 5))  # Return empty tensor on failure

    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Get corresponding label path
        label_path = self._get_label_path(img_path)
        
        # Load labels (YOLO format: class x_center y_center width height)
        labels = self._load_labels(label_path)
        
        # Get original image dimensions
        height, width = img.shape[:2]
        
        # Convert YOLO format to absolute coordinates
        if len(labels) > 0:
            # Convert normalized xywh to absolute xyxy format
            x_center, y_center = labels[:, 1] * width, labels[:, 2] * height
            w, h = labels[:, 3] * width, labels[:, 4] * height
            
            # Create absolute xyxy coordinates (x1, y1, x2, y2)
            x1, y1 = x_center - w/2, y_center - h/2
            x2, y2 = x_center + w/2, y_center + h/2
            
            # Stack into [class_id, x1, y1, x2, y2]
            boxes = torch.stack((labels[:, 0], x1, y1, x2, y2), dim=1)
        else:
            boxes = torch.zeros((0, 5))
        
        # Resize image to model input size (640x640)
        img = cv2.resize(img, (640, 640))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize
        
        # Apply transforms if provided
        if self.transforms:
            img, boxes = self.transforms(img, boxes)
            
        return img, boxes