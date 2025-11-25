import torch
from torch import nn
from torch import device
from torch.cuda.amp import autocast
from torch.nn import functional as F

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        print(f"After ToyModel fc1: {x.dtype}")
        x = self.relu(x)
        print(f"After ToyModel relu: {x.dtype}")
        x = self.ln(x)
        print(f"After ToyModel ln: {x.dtype}")
        x = self.fc2(x)
        print(f"After ToyModel fc2: {x.dtype}")
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ToyModel(64, 32).to(device)
input_data = torch.randn(16, 64).to(device)

for ii in range(1):
    # Forward pass with mixed precision
    model.train()
    with autocast(dtype=torch.float16):
        output = model(input_data)
        loss = F.mse_loss(output, torch.randn(16, 32).to(device))
        # Data type of model parameters within autocast context
        for param in model.parameters():
            print(param.dtype)

        # Data type of output and loss within autocast context
        print(f"Output dtype: {output.dtype}")
        print(f"Loss dtype: {loss.dtype}")

        model.zero_grad()
        loss.backward()
        # Data type of gradients after backward pass
        for param in model.parameters():
            if param.grad is not None:
                print(f"Gradient dtype for param: {param.grad.dtype}")

