# config.py

# List of objects considered hazardous
HAZARDOUS_OBJECTS = [
    'coin', 'drink', 'fork', 'hammer', 'screwdriver', 'stapler', 
    'sharp-item', 'cell phone', 'knife', 'scissor', 'battery'
]

# Distance threshold in meters
DEFAULT_DISTANCE_THRESHOLD = 1.0

# Other configuration parameters can go here
DEFAULT_KNOWN_WIDTH = 0.5
DEFAULT_MINKOWSKI_P = 2