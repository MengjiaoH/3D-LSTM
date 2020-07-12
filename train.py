from utils import h5_virtual_file, window, weights_init
from torch.utils.data import DataLoader
from data_provider import SlidingWindowDataset

taxibj_dir = "./data/TaxiBJ/"

f = h5_virtual_file(
    [
        f"{taxibj_dir}BJ13_M32x32_T30_InOut.h5",
        f"{taxibj_dir}BJ14_M32x32_T30_InOut.h5",
        f"{taxibj_dir}BJ15_M32x32_T30_InOut.h5",
        f"{taxibj_dir}BJ16_M32x32_T30_InOut.h5",
    ]
)

data = f.get("data")

## generate data

batch_size = 20
n_epoches = 2

train_dataset = SlidingWindowDataset.SlidingWindowDataset(data, lambda t: t / 255)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,)
test_dataset = SlidingWindowDataset.SlidingWindowDataset(data, lambda t: t / 255, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,)

for epoch in range(n_epoches):
    for i, (input, target) in enumerate(train_dataloader):
        print("input size", input.size())
        print("output size", target.size())
