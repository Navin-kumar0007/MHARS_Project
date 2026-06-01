"""
MHARS — Stage 6: Federated Learning
===================================
Simulates a Federated Averaging (FedAvg) environment where multiple
edge clients train local models and aggregate them on a central server.
Ensures privacy-preserving fleet learning without sharing raw sensor data.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class FederatedClient:
    def __init__(self, client_id, model, X_data, y_data, batch_size=32, is_v2=True):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.is_v2 = is_v2
        
        if self.is_v2:
            x_tensor = torch.FloatTensor(X_data)
        else:
            x_tensor = torch.FloatTensor(X_data).unsqueeze(-1)
            
        dataset = TensorDataset(x_tensor, torch.FloatTensor(y_data))
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.num_samples = len(X_data)
        
    def train(self, global_state_dict, epochs=5, lr=0.001):
        """Train locally starting from the global weights."""
        self.model.load_state_dict(global_state_dict)
        self.model.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        epoch_loss = 0.0
        for _ in range(epochs):
            for xb, yb in self.loader:
                optimizer.zero_grad()
                preds = self.model(xb)
                if isinstance(preds, tuple): # Handle TFT returning multiple values
                    preds = preds[0] # Quantiles
                
                # If target is 1D but preds is 2D (quantiles), extract median
                if yb.dim() == 1 and preds.dim() == 2 and preds.shape[1] > 1:
                    preds = preds[:, 1] # Median
                    
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
        return self.model.state_dict(), self.num_samples


class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model
        
    def aggregate(self, client_weights_and_samples):
        """
        FedAvg: Weighted average of client model weights.
        client_weights_and_samples: list of (state_dict, num_samples)
        """
        total_samples = sum([n for _, n in client_weights_and_samples])
        
        # Initialize an empty state dict with the same keys as the global model
        aggregated_weights = {k: torch.zeros_like(v) 
                              for k, v in self.global_model.state_dict().items()}
        
        for client_weights, n in client_weights_and_samples:
            weight_factor = n / total_samples
            for k, v in client_weights.items():
                aggregated_weights[k] += v * weight_factor
                
        self.global_model.load_state_dict(aggregated_weights)
        return self.global_model.state_dict()


def run_federated_simulation(global_model, client_datasets, rounds=10, local_epochs=5, is_v2=True):
    """
    Simulates the FedAvg process.
    client_datasets: list of tuples (X_data, y_data)
    """
    print(f"\n[Federated Learning] Starting FedAvg for {rounds} rounds with {len(client_datasets)} clients...")
    
    server = FederatedServer(global_model)
    clients = []
    for i, (X, y) in enumerate(client_datasets):
        clients.append(FederatedClient(f"Client_{i}", global_model, X, y, is_v2=is_v2))
        
    for r in range(rounds):
        print(f"  Round {r+1}/{rounds}")
        global_state = server.global_model.state_dict()
        
        client_updates = []
        for client in clients:
            weights, n_samples = client.train(global_state, epochs=local_epochs)
            client_updates.append((weights, n_samples))
            
        server.aggregate(client_updates)
        print(f"    Aggregated weights from {len(clients)} clients.")
        
    print("[Federated Learning] Simulation complete.\n")
    return server.global_model
