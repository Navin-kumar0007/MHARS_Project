import numpy as np
from typing import Dict, Any, List

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class ExplainabilityEngine:
    """
    Model Explainability (XAI) Engine.
    Computes feature attributions to explain *why* the model made a decision.
    """
    
    def __init__(self):
        pass
        
    def compute_attribution(self, model, input_tensor, method="gradient") -> Dict[str, float]:
        """
        Compute feature attribution for a given model and input.
        
        Args:
            model: PyTorch model (e.g., LSTM or Fusion)
            input_tensor: Input to the model (must require gradients if method="gradient")
            method: Attribution method ("gradient", "integrated_gradients", "fallback")
            
        Returns:
            Dictionary of feature indices to relative importance (0-1).
        """
        if not TORCH_AVAILABLE or method == "fallback":
            return self._fallback_attribution(input_tensor)
            
        if method == "gradient":
            return self._gradient_attribution(model, input_tensor)
            
        return self._fallback_attribution(input_tensor)
        
    def _gradient_attribution(self, model, input_tensor) -> Dict[str, float]:
        """
        Simple Input x Gradient attribution (fast, suitable for real-time).
        """
        # Ensure input requires gradient
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = model(input_tensor)
        
        # We assume output is scalar or we take the mean if multi-dimensional
        if output.numel() > 1:
            loss = output.mean()
        else:
            loss = output
            
        # Backward pass to get gradients w.r.t input
        model.zero_grad()
        loss.backward()
        
        # Input x Gradient
        attr = (input_tensor * input_tensor.grad).detach().cpu().numpy()
        
        # Aggregate over sequence length if 3D tensor (Batch, Seq, Features)
        if attr.ndim == 3:
            # Sum absolute attributions over the sequence length
            attr_sum = np.sum(np.abs(attr[0]), axis=0) 
        elif attr.ndim == 2:
            attr_sum = np.abs(attr[0])
        else:
            attr_sum = np.abs(attr)
            
        # Normalize to sum to 100
        total = np.sum(attr_sum) + 1e-8
        normalized = (attr_sum / total) * 100
        
        # Map to feature names (assuming 5 features for v2 models)
        feature_names = ["LPC_Temp", "HPC_Temp", "Primary_Temp", "HPT_Temp", "LPT_Temp"]
        
        result = {}
        for i, val in enumerate(normalized):
            name = feature_names[i] if i < len(feature_names) else f"Feature_{i}"
            result[name] = round(float(val), 1)
            
        return result
        
    def _fallback_attribution(self, input_tensor) -> Dict[str, float]:
        """
        Fallback attribution when Torch isn't available or model is opaque.
        Just uses variance of the input features as a proxy for importance.
        """
        # Convert to numpy if it's a tensor
        if TORCH_AVAILABLE and isinstance(input_tensor, torch.Tensor):
            arr = input_tensor.detach().cpu().numpy()
        else:
            arr = np.array(input_tensor)
            
        if arr.ndim == 3:
            variance = np.var(arr[0], axis=0)
        elif arr.ndim == 2:
            variance = np.var(arr, axis=0)
        else:
            variance = np.abs(arr) # Just use magnitude if 1D
            
        total = np.sum(variance) + 1e-8
        normalized = (variance / total) * 100
        
        feature_names = ["LPC_Temp", "HPC_Temp", "Primary_Temp", "HPT_Temp", "LPT_Temp"]
        result = {}
        for i, val in enumerate(normalized):
            name = feature_names[i] if i < len(feature_names) else f"Feature_{i}"
            result[name] = round(float(val), 1)
            
        return result
        
    def generate_counterfactual(self, current_temp: float, target_action: str) -> str:
        """
        Generates a simple text counterfactual explanation.
        """
        if target_action in ["fan+", "throttle"]:
            return f"If temperature were 5°C lower, system would remain in 'do-nothing' state."
        elif target_action == "emergency-shutdown":
            return f"If temperature was below critical threshold, system would attempt 'throttle' instead."
        return "System is in optimal state."
