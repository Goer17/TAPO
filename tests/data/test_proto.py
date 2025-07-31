from verl.protocol import *
import torch

b = DataProto.from_single_dict(
    {
        "x": torch.tensor([[1, 2], [3, 4]]).float(),
        "y": torch.tensor([1, 0]).float()
    }
)

print(b.batch)

lst = b.chunk(2)

for _b in lst:
    print(_b.batch)