#!/usr/bin/env python3
"""
Simple test script for Neural ODE implementation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

print("=== Simple Neural ODE Test ===")

# Load data
df = pd.read_csv('sample_helechos_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f"Data loaded: {df.shape}")

# Simple ODE function
class SimpleODEFunc(nn.Module):
    def __init__(self, input_dim=2):
        super(SimpleODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, input_dim)
        )
    
    def forward(self, t, y):
        return self.net(y)

# Test with simple 2D system
model = SimpleODEFunc(2)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Test data
y0 = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
t = torch.linspace(0, 1, 10)

print("Testing ODE integration...")
try:
    solution = odeint(model, y0, t, method='rk4')
    print(f"Success! Solution shape: {solution.shape}")
    print(f"Initial: {y0[0].numpy()}")
    print(f"Final: {solution[-1][0].numpy()}")
except Exception as e:
    print(f"Error: {e}")

print("Simple NODE test completed!")