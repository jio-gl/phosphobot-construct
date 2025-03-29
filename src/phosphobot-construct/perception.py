"""
Perception and Scene Understanding for the Phosphobot Construct.

This module converts real-world sensor inputs to 3D scene representations
using state-of-the-art vision-language models.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import cv2

logger = logging.getLogger(__name__)

# Import conditional to make the module work even without dependencies
try:
    import torch
    import clip
    HAS_CLIP = True
except ImportError:
    logger.warning("CLIP not installed. Install with: pip install clip")
    HAS_CLIP = False

try:
    from segment_anything import SamPredictor, sam_model_registry
    HAS_SAM = True
except ImportError:
    logger.warning("Segment Anything Model not installed. Install with: pip install segment-anything")
    HAS_SAM = False


class SceneUnderstanding:
    """
    Converting raw sensor data to 3D scene representations.
    
    This class uses vision-language models to understand the scene
    and extract object information.
    """
    
    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        sam_checkpoint: str = "sam_vit_h_4b8939.pth",
        device: str = "cuda"
    ):
        """
        Initialize the scene understanding system.
        
        Args:
            clip_model_name: Name of the CLIP model to use.
            sam_checkpoint: Path to the SAM model checkpoint.
            device: Device to run the models on ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        # Initialize CLIP for object recognition if available
        self.clip_model = None
        self.clip_preprocess = None
        if HAS_CLIP:
            try:
                logger.info(f"Loading CLIP model {clip_model_name} on {self.device}")
                self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
            except Exception as e:
                logger.error(f"Failed to load CLIP model: {str(e)}")
        
        # Initialize SAM for segmentation if available
        self.sam_predictor = None
        if HAS_SAM:
            try:
                logger.info(f"Loading SAM model from {sam_checkpoint} on {self.device}")
                sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
                sam.to(self.device)
                self.sam_predictor = SamPredictor(sam)
            except Exception as e:
                logger.error(f"Failed to load SAM model: {str(e)}")
    
    def segment_objects(self, rgb_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Segment objects in the RGB image.
        
        Args:
            rgb_image: RGB image to segment.
            
        Returns:
            List of dictionaries with segmentation information.
        """
        if not HAS_SAM or self.sam_predictor is None:
            logger.warning("SAM not available. Using fallback object detection.")
            return self._fallback_segmentation(rgb_image)
        
        try:
            # Prepare image for SAM
            if rgb_image.shape[2] == 4:  # RGBA
                rgb_image = rgb_image[:, :, :3]  # Convert to RGB
                
            # Set image in predictor
            self.sam_predictor.set_image(rgb_image)
            
            # Generate masks automatically
            masks_data = self.sam_predictor.generate()
            
            # Process masks to extract objects
            objects = []
            for i, mask_data in enumerate(masks_data):
                mask = mask_data["segmentation"].astype(np.uint8)
                
                # Calculate area and check if mask is large enough
                area = np.sum(mask)
                min_area = 0.01 * rgb_image.shape[0] * rgb_image.shape[1]  # 1% of image
                
                if area < min_area:
                    continue  # Skip small masks
                
                # Find bounding box
                y_indices, x_indices = np.where(mask)
                if len(y_indices) == 0 or len(x_indices) == 0:
                    continue  # Skip empty masks
                    
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                # Extract object patch
                object_patch = rgb_image.copy()
                object_patch[~mask.astype(bool)] = 0
                object_patch = object_patch[y_min:y_max, x_min:x_max]
                
                # Store object information
                objects.append({
                    "id": i,
                    "mask": mask,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "area": area,
                    "patch": object_patch
                })
                
            logger.info(f"Segmented {len(objects)} objects in the image")
            return objects
            
        except Exception as e:
            logger.error(f"Error in segmentation: {str(e)}")
            return self._fallback_segmentation(rgb_image)
    
    def _fallback_segmentation(self, rgb_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Fallback method for segmentation when SAM is not available.
        
        Args:
            rgb_image: RGB image to segment.
            
        Returns:
            List of dictionaries with segmentation information.
        """
        # Simple color-based segmentation as fallback
        objects = []
        
        try:
            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            
            # Define color ranges for common objects (red, green, blue)
            color_ranges = [
                ("red", np.array([0, 100, 100]), np.array([10, 255, 255])),
                ("red2", np.array([170, 100, 100]), np.array([180, 255, 255])),
                ("green", np.array([40, 100, 100]), np.array([80, 255, 255])),
                ("blue", np.array([100, 100, 100]), np.array([140, 255, 255]))
            ]
            
            # Find objects for each color range
            for i, (color_name, lower, upper) in enumerate(color_ranges):
                # Create mask for this color
                mask = cv2.inRange(hsv, lower, upper)
                
                # Apply morphology to clean up the mask
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Process each contour
                for j, contour in enumerate(contours):
                    # Check if contour is large enough
                    area = cv2.contourArea(contour)
                    min_area = 0.005 * rgb_image.shape[0] * rgb_image.shape[1]  # 0.5% of image
                    
                    if area < min_area:
                        continue  # Skip small contours
                    
                    # Create mask for this contour
                    obj_mask = np.zeros_like(mask)
                    cv2.drawContours(obj_mask, [contour], 0, 255, -1)
                    
                    # Find bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Extract object patch
                    object_patch = rgb_image.copy()
                    object_patch[obj_mask == 0] = 0
                    object_patch = object_patch[y:y+h, x:x+w]
                    
                    # Store object information
                    objects.append({
                        "id": len(objects),
                        "mask": obj_mask,
                        "bbox": [x, y, x+w, y+h],
                        "area": area,
                        "patch": object_patch,
                        "color": color_name
                    })
            
            logger.info(f"Fallback segmentation found {len(objects)} objects")
            return objects
            
        except Exception as e:
            logger.error(f"Error in fallback segmentation: {str(e)}")
            return []
    
    def classify_objects(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify segmented objects using CLIP.
        
        Args:
            objects: List of dictionaries with object patches.
            
        Returns:
            Updated list of dictionaries with classification information.
        """
        if not HAS_CLIP or self.clip_model is None:
            logger.warning("CLIP not available. Using fallback classification.")
            return self._fallback_classification(objects)
        
        try:
            # Prepare candidate labels for common tabletop objects
            candidate_labels = [
                "a small box", "a medium box", "a large box",
                "a red cube", "a blue cube", "a green cube",
                "a wooden block", "a plastic toy", "a metal container",
                "a cup", "a plate", "a bowl"
            ]
            
            # Encode candidate labels
            text_tokens = clip.tokenize(candidate_labels).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Process each object
            for obj in objects:
                patch = obj["patch"]
                
                # Check if patch is empty
                if patch.size == 0 or np.all(patch == 0):
                    obj["class"] = "unknown"
                    obj["confidence"] = 0.0
                    continue
                
                # Preprocess image for CLIP
                image = self.clip_preprocess(Image.fromarray(patch)).unsqueeze(0).to(self.device)
                
                # Encode image
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # Get the best matching class
                values, indices = similarity[0].topk(3)
                
                # Store classification results
                obj["class"] = candidate_labels[indices[0].item()]
                obj["confidence"] = values[0].item()
                obj["top_classes"] = [
                    {"class": candidate_labels[idx.item()], "confidence": val.item()}
                    for val, idx in zip(values, indices)
                ]
            
            logger.info(f"Classified {len(objects)} objects using CLIP")
            return objects
            
        except Exception as e:
            logger.error(f"Error in object classification: {str(e)}")
            return self._fallback_classification(objects)
    
    def _fallback_classification(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback method for classification when CLIP is not available.
        
        Args:
            objects: List of dictionaries with object patches.
            
        Returns:
            Updated list of dictionaries with classification information.
        """
        # Simple color and size based classification
        for obj in objects:
            # Determine object size based on area
            area_ratio = obj["area"] / (obj["patch"].shape[0] * obj["patch"].shape[1])
            
            if area_ratio > 0.3:
                size = "large"
            elif area_ratio > 0.1:
                size = "medium"
            else:
                size = "small"
            
            # Use color if available, otherwise use size only
            if "color" in obj:
                obj["class"] = f"a {size} {obj['color']} object"
            else:
                # Try to determine dominant color
                if obj["patch"].size > 0:
                    mean_color = np.mean(obj["patch"], axis=(0, 1))
                    if mean_color[0] > mean_color[1] and mean_color[0] > mean_color[2]:
                        color = "red"
                    elif mean_color[1] > mean_color[0] and mean_color[1] > mean_color[2]:
                        color = "green"
                    elif mean_color[2] > mean_color[0] and mean_color[2] > mean_color[1]:
                        color = "blue"
                    else:
                        color = "gray"
                    
                    obj["class"] = f"a {size} {color} object"
                else:
                    obj["class"] = f"a {size} object"
            
            obj["confidence"] = 0.7  # Arbitrary confidence for fallback
        
        logger.info(f"Fallback classification for {len(objects)} objects")
        return objects
    
    def estimate_3d_positions(
        self, 
        objects: List[Dict[str, Any]], 
        depth_image: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Estimate 3D positions of objects using depth information.
        
        Args:
            objects: List of dictionaries with object information.
            depth_image: Depth image corresponding to the RGB image.
            
        Returns:
            Updated list of dictionaries with 3D position information.
        """
        # Get image dimensions
        if len(objects) > 0 and "mask" in objects[0]:
            h, w = objects[0]["mask"].shape[:2]
        else:
            return objects
        
        # Process each object
        for obj in objects:
            # Extract bounding box
            x_min, y_min, x_max, y_max = obj["bbox"]
            
            # Calculate center in image coordinates
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            
            # Convert to normalized coordinates (-1 to 1)
            norm_x = (center_x / w) * 2 - 1
            norm_y = (center_y / h) * 2 - 1
            
            # Estimate depth
            if depth_image is not None:
                # Use depth information from depth image
                depth_region = depth_image[y_min:y_max, x_min:x_max]
                if depth_region.size > 0:
                    # Use median depth to be robust to outliers
                    depth = np.median(depth_region[depth_region > 0])
                else:
                    depth = 1.0  # Default depth if region is empty
            else:
                # Estimate depth from y-position (objects lower in image are closer)
                # This is a very rough estimate based on perspective
                depth = 1.0 + 0.5 * (1 - norm_y)
            
            # Store 3D position (x, y, z) in world coordinates
            # Note: This is a simplified mapping and would need camera calibration
            # for accurate results
            obj["position_3d"] = {
                "x": norm_x * depth,
                "y": norm_y * depth,
                "z": depth
            }
        
        logger.info(f"Estimated 3D positions for {len(objects)} objects")
        return objects


def perception_pipeline(
    rgb_image: np.ndarray,
    depth_image: Optional[np.ndarray] = None,
    proprioception: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Convert raw sensor data to 3D scene representation (the 'placing' operation).
    
    Args:
        rgb_image: RGB image from the camera.
        depth_image: Depth image from the camera (optional).
        proprioception: Robot state vector (optional).
        
    Returns:
        Dictionary with 3D scene representation.
    """
    logger.info("Starting perception pipeline")
    
    # Initialize scene understanding system
    scene_system = SceneUnderstanding(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Segment objects in the image
    objects = scene_system.segment_objects(rgb_image)
    
    # 2. Classify the objects
    objects = scene_system.classify_objects(objects)
    
    # 3. Estimate 3D positions
    objects = scene_system.estimate_3d_positions(objects, depth_image)
    
    # 4. Create 3D scene representation
    scene_3d = {
        "objects": objects,
        "workspace": {
            "width": rgb_image.shape[1],
            "height": rgb_image.shape[0],
            "depth": np.max(depth_image) if depth_image is not None else 2.0
        }
    }
    
    # Add robot state if available
    if proprioception is not None:
        scene_3d["robot_state"] = proprioception.tolist()
    
    logger.info(f"Created 3D scene with {len(objects)} objects")
    return scene_3d


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test the perception pipeline with a sample image
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # Load sample image
    try:
        sample_path = "data/samples/scene.jpg"
        sample_image = np.array(Image.open(sample_path))
        
        # Run perception pipeline
        scene = perception_pipeline(sample_image)
        
        # Visualize results
        plt.figure(figsize=(12, 8))
        plt.imshow(sample_image)
        
        # Overlay bounding boxes and labels
        for obj in scene["objects"]:
            x_min, y_min, x_max, y_max = obj["bbox"]
            plt.plot([x_min, x_max, x_max, x_min, x_min], 
                     [y_min, y_min, y_max, y_max, y_min], 'r-')
            plt.text(x_min, y_min - 5, f"{obj['class']} ({obj['confidence']:.2f})")
        
        plt.title("Scene Understanding Results")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logger.error(f"Error in sample processing: {str(e)}")