from utils import h5_virtual_file, window, weights_init
import torch
from torch.utils.data import DataLoader
from data_provider import SlidingWindowDataset
from model.model import LSTM_Model

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
n_epoches = 5
window_length = 2
window_stride = 1
input_length = 4
output_length = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = SlidingWindowDataset.SlidingWindowDataset(data, lambda t: t / 255)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,)
test_dataset = SlidingWindowDataset.SlidingWindowDataset(data, lambda t: t / 255, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,)

# Model 
model = LSTM_Model()

for epoch in range(n_epoches):
    for i, (input, target) in enumerate(train_dataloader):
        # print("input size", input.size())
        # print("output size", target.size())
        frames = []
        for indices in window(range(input_length), window_length, window_stride):
            frames.append(input[:, :, indices[0] : indices[-1] + 1])
        input = torch.stack(frames, dim=0).to(device)
        target = target.to(device)
        model.to(device)
        model.train()
        model.optimizer.zero_grad()
        l1_loss, l2_loss = model.loss(input, target)
        loss = l1_loss + l2_loss
        loss.backward()
        model.optimizer.step()

        if i % 1 == 0:
            print("Epoch: {}/{}, step: {}/{}, mse: {}".format(epoch, n_epoches, i, len(train_dataloader), l2_loss))
