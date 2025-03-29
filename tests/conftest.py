import sys
import os

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Adding to path: {parent_dir}")
print(f"Current sys.path: {sys.path}")
sys.path.insert(0, parent_dir)
print(f"Updated sys.path: {sys.path}")