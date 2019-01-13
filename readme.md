# pytorch-utilities
This repo contains commonly used functions to help me move faster. For example, writing a training loop, testing and calculating metrics for them are all things that are done quite often. I wrote a BaseNetwork abstraction to handle these to save me time. 

There are probably higher level libraries out there that do the very same thing, but it's more fun to DIY.

For example, to perform cross validation:
1) Complete the templates in the templates/ folder (and put them into project root dir)
2) Run a training script something like this:
```
import dataset
import net
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from utilities.evaluate_helpers import cross_validate

params = {
    'batch_size': 100,
    'shuffle': True, 
    'num_workers': 0,
}

device = torch.device('cuda:0')
epochs = 10
learning_rate = .0001
cross_validation_folds = 10

your_dataset = dataset.YourDataset()
model = net.Net(285, nn.MSELoss()).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.set_optimizer(optimizer)

cross_validate(model, your_dataset, 10, 10, params)
```
Possible output (depending on how you completed the templates):
```
Cross validation fold 0
[TRAIN Epoch 0][Accuracy:     0.3429][MSE Loss:     0.2542]
[TRAIN Epoch 1][Accuracy:     0.3429][MSE Loss:     0.2537]
[TRAIN Epoch 2][Accuracy:     0.3429][MSE Loss:     0.2533]
[TRAIN Epoch 3][Accuracy:     0.3429][MSE Loss:     0.2530]
[TRAIN Epoch 4][Accuracy:     0.3429][MSE Loss:     0.2526]
[TRAIN Epoch 5][Accuracy:     0.3429][MSE Loss:     0.2522]
[TRAIN Epoch 6][Accuracy:     0.3429][MSE Loss:     0.2518]
[TRAIN Epoch 7][Accuracy:     0.3714][MSE Loss:     0.2515]
[TRAIN Epoch 8][Accuracy:     0.3714][MSE Loss:     0.2511]
[TRAIN Epoch 9][Accuracy:     0.4286][MSE Loss:     0.2507]

Test with dataset of size 5
TEST[Accuracy:     0.4000][MSE Loss:     0.2511]

...

########################################
Cross validation with 10 folds complete.
Average Accuracy:     0.6367
Average MSE Loss:     0.2240
```
