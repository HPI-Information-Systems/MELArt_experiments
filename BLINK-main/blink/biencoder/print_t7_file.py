import torch
import torchfile

data = torch.load('/scratch/user/uqlle6/code/artemo/BLINK-main/models/artel_wcat/top64_candidates/test.t7')

# Convert the loaded data to a PyTorch tensor if applicable
# tensor_data = torch.tensor(data)

print(data[0])