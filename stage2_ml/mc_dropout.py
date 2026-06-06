import numpy as np
from typing import Dict, Any, List

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class MCDropoutEstimator:
    """
    Monte Carlo Dropout Engine for Epistemic Uncertainty Estimation.
    Runs multiple forward passes with dropout enabled to gauge model uncertainty.
    """
    
    def __init__(self, num_samples: int = 30):
        self.num_samples = num_samples
        
    def predict_with_uncertainty(self, model, input_tensor) -> Dict[str, float]:
        """
        Runs Monte Carlo dropout prediction.
        
        Args:
            model: PyTorch model (e.g., ThermalLSTMv2)
            input_tensor: Input to the model
            
        Returns:
            Dict containing mean prediction, variance, std, and confidence [0-100].
        """
        if not TORCH_AVAILABLE:
            return self._fallback_prediction(input_tensor)
            
        # Ensure model is in training mode to enable dropout
        model.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = model(input_tensor)
                # If output is a tuple (like from TFT), take the median prediction
                if isinstance(output, tuple):
                    predictions.append(output[0][0, 1].item())
                else:
                    predictions.append(output.item())
                    
        # Restore to eval mode
        model.eval()
        
        preds = np.array(predictions)
        mean_pred = float(np.mean(preds))
        variance = float(np.var(preds))
        std_dev = float(np.std(preds))
        
        # Calculate a 0-100 confidence score based on variance
        # High variance = low confidence
        # Heuristic: 0.1 variance is quite high for normalized temps [0,1]
        confidence = max(0.0, min(100.0, 100.0 * (1.0 - (variance * 10.0))))
        
        # 95% Confidence Interval (approximate)
        lower_bound = float(np.percentile(preds, 2.5))
        upper_bound = float(np.percentile(preds, 97.5))
        
        return {
            "mean": mean_pred,
            "variance": variance,
            "std": std_dev,
            "confidence": round(confidence, 1),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
        
    def _fallback_prediction(self, input_tensor) -> Dict[str, float]:
        """Fallback if Torch is not available."""
        # We can't really do MCDropout without Torch, return dummies
        if isinstance(input_tensor, list):
            arr = np.array(input_tensor)
            val = float(np.mean(arr[-1]))
        elif hasattr(input_tensor, "cpu"):
            arr = input_tensor.detach().cpu().numpy()
            val = float(np.mean(arr[0, -1]))
        else:
            val = 0.5
            
        return {
            "mean": val,
            "variance": 0.05,
            "std": 0.22,
            "confidence": 85.0,
            "lower_bound": val - 0.05,
            "upper_bound": val + 0.05
        }
