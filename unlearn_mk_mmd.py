import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from datetime import datetime
from main_bert import TimeSeriesBERTFineTune
from data_loader import load_test_data
import utils2
import matplotlib.pyplot as plt

from mk_mmd import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel
# Import the fixed MK-MMD implementation
from typing import Optional, Sequence

class UnlearningModelWithMKMMD:
    """
    Batched MK-MMD Enhanced Unlearning Model - accumulates losses over batches before unlearning
    """
    
    def __init__(self, model, batch_size=None):
        self.model = model
        self.batch_size = batch_size
        
        # Hardcoded parameters matching the original code
        self.THRESHOLD  = 0.07
        self.BND = 0.07 * 2  # BND is double the threshold
        self.LAMB = 0
        self.UNLEARN_LR = 1e-5
        self.UNLEARN_EPOCHS = 10
        self.HISTORY = 100
        self.WINDOW = 90 # Fixed window for cached detection
        self.device = next(model.parameters()).device
        
        # MK-MMD parameters
        self.MKMMD_WEIGHT = 1# Weight for MK-MMD term
        self.USE_MKMMD_BOUND = True  # Whether to use MK-MMD in bounding loss
        
        # Initialize MK-MMD with multiple Gaussian kernels
        self.mkmmd = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[
                GaussianKernel(alpha=2**3),
                GaussianKernel(alpha=2**2),
                GaussianKernel(alpha=2**4),
                GaussianKernel(alpha=2**5),
                GaussianKernel(alpha=2**1)
            ], linear= True)

    def batch_unlearn_step_mkmmd(self, batch_sequences, batch_targets, batch_labels):
        """
        Batched MK-MMD unlearning step - processes multiple samples together with MK-MMD enhancement
        
        Args:
            batch_sequences: Tensor of shape [batch_size, seq_len, features]
            batch_targets: Tensor of shape [batch_size, features] or [batch_size, seq_len, features]
            batch_labels: List/array of labels for each sample in batch
        """
        self.model.train()
        
        # Convert labels to loss multipliers
        lt_batch = torch.tensor([label * 2 - 1 for label in batch_labels], 
                            dtype=torch.float32, device=self.device)  # +1: FN, -1: FP
        
        old_params = torch.cat([p.detach().view(-1) for p in self.model.parameters()])
        
        criterion = nn.MSELoss(reduction="none")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.UNLEARN_LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01)
        
        for epoch in range(self.UNLEARN_EPOCHS):
            # Get predictions for all samples in batch
            batch_preds = self.model(batch_sequences)
            
            # Calculate loss based on actual tensor shapes
            if batch_preds.shape == batch_targets.shape:
                # Direct comparison - same shapes
                loss_per_sample = criterion(batch_preds, batch_targets)
                # Average over all dimensions except batch dimension
                loss_vec = loss_per_sample.view(batch_preds.shape[0], -1).mean(dim=1)
            
            elif len(batch_preds.shape) == 3 and len(batch_targets.shape) == 2:
                # Predictions: [batch_size, seq_len, features], Targets: [batch_size, features]
                # Use last timestep of predictions
                batch_preds_last = batch_preds[:, -1, :]
                loss_per_sample = criterion(batch_preds_last, batch_targets)
                loss_vec = loss_per_sample.mean(dim=1)  # Average over features
            
            elif len(batch_preds.shape) == 2 and len(batch_targets.shape) == 2:
                # Both: [batch_size, features]
                loss_per_sample = criterion(batch_preds, batch_targets)
                loss_vec = loss_per_sample.mean(dim=1)  # Average over features
            
            else:
                # Flatten both and compare
                batch_preds_flat = batch_preds.view(batch_preds.shape[0], -1)
                batch_targets_flat = batch_targets.view(batch_targets.shape[0], -1)
                
                # Ensure they have the same number of elements
                min_size = min(batch_preds_flat.shape[1], batch_targets_flat.shape[1])
                batch_preds_flat = batch_preds_flat[:, :min_size]
                batch_targets_flat = batch_targets_flat[:, :min_size]
                
                loss_per_sample = criterion(batch_preds_flat, batch_targets_flat)
                loss_vec = loss_per_sample.mean(dim=1)
            
            # Batched MK-MMD computation with enhanced strategies
            mkmmd_values = []
            
            for i in range(batch_sequences.shape[0]):
                try:
                    # Strategy 1: Use sequence timesteps as different samples for each batch item
                    seq_for_mkmmd = batch_sequences[i]  # Shape: (seq_len, features)
                    target_for_mkmmd = batch_targets[i].unsqueeze(0)  # Shape: (1, features)
                    
                    # Ensure minimum samples for MK-MMD
                    if seq_for_mkmmd.shape[0] < 2:
                        # If sequence too short, duplicate target
                        seq_expanded = target_for_mkmmd.repeat(2, 1)
                        mkmmd_value = self.mkmmd(seq_for_mkmmd, seq_expanded)
                    else:
                        mkmmd_value = self.mkmmd(seq_for_mkmmd, target_for_mkmmd)
                    
                    mkmmd_values.append(mkmmd_value)
                    
                except Exception as e:
                    try:
                        # Strategy 2: Compare sequence statistics vs target
                        seq_mean = batch_sequences[i].mean(dim=0, keepdim=True)  # (1, features)
                        seq_std = batch_sequences[i].std(dim=0, keepdim=True)    # (1, features)
                        sequence_stats = torch.cat([seq_mean, seq_std], dim=0)
                        target_stats = batch_targets[i].unsqueeze(0).repeat(2, 1)
                        mkmmd_value = self.mkmmd(sequence_stats, target_stats)
                        mkmmd_values.append(mkmmd_value)
                        
                    except Exception as e2:
                        try:
                            # Strategy 3: Error-based approach
                            pred_error = loss_vec[i].unsqueeze(0).unsqueeze(1)  # [1, 1]
                            target_error = torch.zeros_like(pred_error, device=self.device)
                            # Create minimum samples for MK-MMD
                            pred_error_expanded = pred_error.repeat(2, 1)
                            target_error_expanded = target_error.repeat(2, 1)
                            mkmmd_value = self.mkmmd(pred_error_expanded, target_error_expanded)
                            mkmmd_values.append(mkmmd_value)
                        except Exception as e3:
                            # Final fallback
                            mkmmd_values.append(torch.tensor(self.BND, device=self.device))
            
            # Convert MK-MMD values to tensor
            mkmmd_tensor = torch.stack(mkmmd_values)
            
            # Apply MK-MMD-enhanced bounding loss for each sample
            if self.USE_MKMMD_BOUND:
                # Use MK-MMD values instead of fixed BND
                bounding_losses = torch.relu(torch.abs(mkmmd_tensor * self.MKMMD_WEIGHT) - lt_batch * loss_vec)
            else:
                # Fall back to original BND approach
                bounding_losses = torch.relu(self.BND - lt_batch * loss_vec)
            
            total_bounding_loss = torch.sum(bounding_losses)
            
            # Parameter regularization
            current_params = torch.cat([p.view(-1) for p in self.model.parameters()])
            reg_loss = 0.5 * self.LAMB * (current_params - old_params).norm(2) ** 2
            
            total_loss = total_bounding_loss  + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        return self.model

    def unlearn_on_test_data(self, Xtest, Ytest):
        """
        Apply batched MK-MMD unlearning on test data
        """
        self.model.eval()
        
        # Collect data for batching
        batch_sequences = []
        batch_targets = []
        batch_labels = []
        
        for i in range(self.HISTORY, len(Xtest)):
            sequence = Xtest[i - self.HISTORY : i]
            target = Xtest[i]
            label = int(Ytest[i])
            
            sequence_tensor = torch.tensor(
                sequence, dtype=torch.float32, device=self.device
            )
            
            # Get initial prediction to check if unlearning is needed
            with torch.no_grad():
                pred = self.model(sequence_tensor.unsqueeze(0)).detach().cpu().numpy().squeeze()
                error = np.mean((pred - target) ** 2)
            
            initial_anomaly = int(error > self.THRESHOLD)
            needs_unlearning = (initial_anomaly != label)
            
            # Add to batch if unlearning is needed
            if needs_unlearning:
                batch_sequences.append(sequence_tensor)
                # Ensure target is always a 1D tensor representing the target timestep
                target_tensor = torch.tensor(target, dtype=torch.float32, device=self.device)
                if len(target_tensor.shape) == 0:  # scalar
                    target_tensor = target_tensor.unsqueeze(0)
                batch_targets.append(target_tensor)
                batch_labels.append(label)
            
            # Process batch when it's full
            if len(batch_sequences) >= self.batch_size:
                # Stack tensors for batch processing
                batch_seq_tensor = torch.stack(batch_sequences)  # [batch_size, seq_len, features]
                batch_target_tensor = torch.stack(batch_targets)  # [batch_size, features]
                
                # Apply batched MK-MMD unlearning
                self.model = self.batch_unlearn_step_mkmmd(batch_seq_tensor, batch_target_tensor, batch_labels)
                
                # Clear batch
                batch_sequences = []
                batch_targets = []
                batch_labels = []
        
        # Process remaining samples
        if len(batch_sequences) > 0:
            batch_seq_tensor = torch.stack(batch_sequences)
            batch_target_tensor = torch.stack(batch_targets)
            self.model = self.batch_unlearn_step_mkmmd(batch_seq_tensor, batch_target_tensor, batch_labels)
        
        return self.model


# Example usage
if __name__ == "__main__":
    from main_bert import TimeSeriesBERTFineTune
    from data_loader import load_test_data
    
    # Load data
    dataset_name = "WADI-CLEAN"
    Xtest, Ytest, _ = load_test_data(dataset_name)
    
    # Parameters
    input_dim = 110
    output_dim = 110
    history = 100
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesBERTFineTune(
        input_dim=input_dim, 
        output_dim=output_dim, 
        seq_len=history, 
        freeze_until_layer=12
    )
    model.load_state_dict(torch.load("Final_SecBert_W.pt"))
    model.to(device)
    
    # Apply unlearning
    batch_size = 8
    unlearner = UnlearningModelWithMKMMD(model, batch_size=batch_size)
    unlearned_model = unlearner.unlearn_on_test_data(Xtest, Ytest)
    
    # Save unlearned model
    torch.save(unlearned_model.state_dict(), "unlearned_model.pt")
    print("Unlearning complete! Model saved.")