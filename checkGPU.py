# import torch
# print("CUDA Available:", torch.cuda.is_available())
# print("CUDA Device Count:", torch.cuda.device_count())
# print("Using GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

import torch

print("PyTorch Version:", torch.__version__)
print("MPS Available:", torch.backends.mps.is_available())
print("MPS Built:", torch.backends.mps.is_built())
