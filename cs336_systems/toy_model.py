import torch.nn as nn
import torch


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print(f"fc1 type: {x.dtype}")
        x = self.ln(x)
        print(f"ln type: {x.dtype}")
        x = self.fc2(x)
        print(f"fc2 type: {x.dtype}")
        return x


model = ToyModel(6, 2)
x = torch.Tensor([1, 2, 3, 4, 5, 6])
with torch.autocast(device_type="mps", dtype=torch.float16):
    y = model(x)
