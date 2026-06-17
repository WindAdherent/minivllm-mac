import torch

print("torch:", torch.__version__)
print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())

device = "mps" if torch.backends.mps.is_available() else "cpu"
x = torch.randn(2, 3, device=device)
print(device, x @ x.T)