import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utilities.base_network import BaseNetwork
import utilities.metrics as metrics


class Net(BaseNetwork):

    def __init__(self, optimizer, criterion):
        super(Net, self).__init__()
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric_to_fn = {}
        self.metric_to_value_sums = {}
        """
        TODO: define stuff here

        E.g
        super(Net, self).__init__()
        self.metric_to_fn = {
            # weighted by number of samples in each batch. E.g accuracy weighted by
            # number of samples is number of correct predictions.
            'Accuracy': metrics.correct_predictions_two_class,
            'MSE Loss': nn.MSELoss(reduction='sum')
        }
        self.metric_to_value_sums = {}
        self.criterion = criterion
        self.optimizer = optimizer

        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 1)
        """


    def forward(self, x):
        """
        TODO: define how input passes through layers.
        E.g
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x
        """

