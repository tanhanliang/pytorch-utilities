import torch
import torch.nn as nn
import warnings


class BaseNetwork(nn.Module):

    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.metric_to_fn = {}
        self.metric_to_value_sums = {}
        self.criterion = None
        self.optimizer = None


    # Old school way of making this class abstract
    def forward(self, x):
        raise NotImplementedError("The forward() method has not been implemented.")


    def train_batch(self, inputs, targets):
        """
        Trains the model over one batch, and applies one step of gradient descent.
        @param inputs: inputs to the network as a torch.Tensor
        @param targets: what you want the network to output, as a torch.Tensor
        """
        if self.criterion == None or self.optimizer == None:
            raise NotImplementedError("The criterion or optimizer has not been set.")

        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        # remove from computation graph to prevent retention of gradients after train step
        outputs = outputs.detach()
        targets = targets.detach()
        self.compute_metrics(outputs, targets)


    def compute_metrics(self, outputs, targets):
        """
        Compute metrics that are defined in the child class. 
        Weights the value of each metric by the number of outputs.

        @param outputs: the outputs of the network as a torch.Tensor
        @param targets: what you want the network to output, as a torch.Tensor
        """
        if len(self.metric_to_fn) == 0:
            warnings.warn('No metrics have been defined.')
            return

        outputs_len = outputs.size()[0]
        for metric, metric_fn in self.metric_to_fn.items():
            self.metric_to_value_sums[metric] += metric_fn(outputs, targets)*outputs_len


    def train_epoch(self, training_generator, epoch):
        """
        Trains the network over one epoch.

        @param training_generator: torch.utils.data.DataLoader. Contains training data
        @param epoch: which epoch this is
        """

        for metric in self.metric_to_fn:
            self.metric_to_value_sums[metric] = 0

        for inputs, targets in training_generator:
            self.train_batch(inputs, targets)

        train_examples = training_generator.dataset.__len__()
        metric_string = '[TRAIN Epoch %d]' % epoch

        for metric, value_sum in self.metric_to_value_sums.items():
            avg_value = value_sum/train_examples
            metric_string += '[{0}: {1:10.4f}]'.format(metric, avg_value) 

        print(metric_string)


    def test(self, inputs, targets):
        """
        Tests the network on some data.
        @param inputs: inputs to the network as a torch.Tensor
        @param targets: what you want the network to output, as a torch.Tensor
        @return: A dictionary containing values of each metric
        """

        outputs = self(inputs)
        dataset_size = targets.size()[0]
        print('\nTest with dataset of size %d' % dataset_size)
        metric_string = 'TEST'
        metric_to_values = {}

        for metric, metric_fn in self.metric_to_fn.items():
            avg_value = metric_fn(outputs, targets)/dataset_size
            metric_string += '[{0}: {1:10.4f}]'.format(metric, avg_value)
            metric_to_values[metric] = avg_value

        print(metric_string)
        return metric_to_values


    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


    def reset_all_weights(self):
        def reset_layer_weights(layer):
            """
            Initialise weights in a layer to a random value sampled from a uniform distribution.
            @param layer: A torch.nn layer in the network.
            """
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        self.apply(reset_layer_weights)
