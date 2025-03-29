"""
Directly modifies the Python path for testing.
Run this before running your tests.
"""
import sys
import os
import site

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

# Print the Python path for debugging
print("\nCurrent Python path:")
for path in sys.path:
    print(f"  - {path}")

# Verify if the module can be found
try:
    import src.phosphobot_construct
    print("\nSuccess! 'src.phosphobot_construct' module was found")
except ImportError as e:
    print(f"\nError: {e}")
    print("\nTroubleshooting steps:")
    print("1. Check if 'src/phosphobot_construct/__init__.py' exists")
    print("2. If your module is in a different location, update your imports")
    
    # Check if the directory exists without being a package
    if os.path.isdir(os.path.join(project_root, 'src', 'phosphobot_construct')):
        print("\nThe directory 'src/phosphobot_construct' exists but might be missing an __init__.py file")
        
    # Check alternative structure
    if os.path.isdir(os.path.join(project_root, 'phosphobot_construct')):
        print("\nFound 'phosphobot_construct' directory directly in project root.")
        print("Try changing imports from 'src.phosphobot_construct' to 'phosphobot_construct'")