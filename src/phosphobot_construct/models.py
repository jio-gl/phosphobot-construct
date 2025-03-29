"""
Phosphobot Construct models for robot control.

This module contains the implementation of neural network models that can
control the phosphobot robot arm based on visual and proprioceptive input.
"""

import numpy as np
import torch
import torch.nn as nn
from phosphobot.am import ActionModel
from typing import Dict, List, Optional, Union, Any


class PhosphoConstructModel(ActionModel):
    """
    A PyTorch model for generating robot actions from robot state, camera images, and text prompts.
    
    This model inherits from the phosphobot ActionModel base class and implements
    a transformer-based architecture for learning from multimodal inputs.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize the PhosphoConstructModel.
        
        Args:
            model_path: Path to pre-trained model weights. If None, model is initialized with random weights.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        # Initialize the model architecture
        self.vision_encoder = self._build_vision_encoder()
        self.state_encoder = self._build_state_encoder()
        self.transformer = self._build_transformer()
        self.action_head = self._build_action_head()
        
        # Load pre-trained weights if provided
        if model_path:
            self.load_model(model_path)
        
        # Move model to device
        self.to(self.device)
        
    def _build_vision_encoder(self) -> nn.Module:
        """Build the vision encoder component."""
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU()
        )
    
    def _build_state_encoder(self) -> nn.Module:
        """Build the proprioception encoder component."""
        return nn.Sequential(
            nn.Linear(6, 128),  # Assuming 6 DoF robot
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU()
        )
    
    def _build_transformer(self) -> nn.Module:
        """Build the transformer component for sequential decision making."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=6)
    
    def _build_action_head(self) -> nn.Module:
        """Build the action prediction head."""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6)  # Output dimension matches the robot's DoF
        )
    
    def load_model(self, model_path: str) -> None:
        """
        Load model weights from file.
        
        Args:
            model_path: Path to the model weights file.
        """
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        self.eval()
    
    def save_model(self, model_path: str) -> None:
        """
        Save model weights to file.
        
        Args:
            model_path: Path to save the model weights.
        """
        torch.save(self.state_dict(), model_path)
    
    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Prepare and normalize the inputs for the model.
        
        Args:
            inputs: Dictionary containing state, images, and optional prompt.
            
        Returns:
            Dict of tensors ready for model input.
        """
        # Process robot state
        state = torch.tensor(inputs["state"], dtype=torch.float32).to(self.device)
        
        # Process images
        images = inputs["images"]
        if isinstance(images, np.ndarray):
            # Normalize pixel values to [0, 1]
            images = images.astype(np.float32) / 255.0
            images = torch.tensor(images).to(self.device)
        elif isinstance(images, torch.Tensor):
            images = images.float().to(self.device) / 255.0
        
        # Ensure images are in the right format [batch, channels, height, width]
        if images.dim() == 3:  # Single image [C, H, W]
            images = images.unsqueeze(0)
        elif images.dim() == 4 and images.shape[1] != 3:  # [N, H, W, C] format
            images = images.permute(0, 3, 1, 2)  # Convert to [N, C, H, W]
            
        return {"state": state, "images": images}
    
    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            inputs: Dictionary with processed inputs.
            
        Returns:
            Tensor of predicted actions.
        """
        # Extract inputs
        state = inputs["state"]
        images = inputs["images"]
        
        # Process multiple images if provided
        batch_size = images.shape[0]
        
        # Encode each image and average the features
        image_features = torch.zeros(batch_size, 512, device=self.device)
        for i in range(batch_size):
            image_features[i] = self.vision_encoder(images[i].unsqueeze(0))
            
        # Encode robot state
        state_features = self.state_encoder(state)
        
        # Combine features
        combined_features = image_features + state_features
        
        # Add sequence dimension for transformer and transpose
        # [batch_size, sequence_length, feature_dim] -> [sequence_length, batch_size, feature_dim]
        combined_features = combined_features.unsqueeze(0).transpose(0, 1)
        
        # Pass through transformer
        transformer_output = self.transformer(combined_features)
        
        # Take output of last sequence element and predict actions
        actions = self.action_head(transformer_output[-1])
        
        return actions
    
    def sample_actions(self, inputs: Dict[str, Any]) -> np.ndarray:
        """
        Select a sequence of actions based on the inputs.
        
        Args:
            inputs: Dictionary with keys:
                - "state": Tensor or list of floats representing robot state.
                - "images": List of images (numpy arrays or tensors).
                - "prompt": String text prompt (optional).
                
        Returns:
            np.ndarray: Sequence of actions (shape: [max_seq_length, n_actions]).
        """
        # Prepare inputs and run forward pass
        with torch.no_grad():
            processed_inputs = self.prepare_inputs(inputs)
            actions = self.forward(processed_inputs)
            
        # Convert to numpy array
        actions_np = actions.cpu().numpy()
        
        # Create a sequence of actions (here just a single action repeated)
        # In a more sophisticated model, you would generate a sequence of actions
        seq_length = 10  # Number of steps to look ahead
        action_sequence = np.tile(actions_np, (seq_length, 1))
        
        return action_sequence
    
    def __call__(self, inputs: Dict[str, Any]) -> np.ndarray:
        """
        Makes the model instance callable, delegating to the sample_actions method.
        
        Args:
            inputs: Dictionary with model inputs.
            
        Returns:
            np.ndarray: Sequence of actions.
        """
        return self.sample_actions(inputs)