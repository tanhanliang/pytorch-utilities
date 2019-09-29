"""
Things to help me evaluate the model.
"""

from sklearn.model_selection import StratifiedKFold 
from torch.utils import data
import torch


def cross_validate(model, dataset, splits, epochs, dataloader_params):
    """
    Does cross validation for a model.

    @param model: An instance of a model to be evaluated.
    @param dataset: A torch.utils.data.Dataset dataset.
    @param splits: The number of cross validation folds.
    @param epochs: The number of epochs per cross validation fold.
    @dataloader_params: parameters to be passed to the torch.utils.data.DataLoader class.
    Typically
    params = {
        'batch_size': 100,
        'shuffle': True, 
        'num_workers': 4,
    }
    """
    skf = StratifiedKFold(n_splits=splits)
    metrics_to_avg_values = {}
    fold = 0

    for train_idx, test_idx in skf.split(dataset.targets, dataset.targets):
        print("\nCross validation fold %d" %fold)

        model.reset_all_weights()
        dataset.set_active_data(train_idx)
        train_generator = data.DataLoader(dataset, **dataloader_params)

        for epoch in range(epochs):
            model.train_epoch(train_generator, epoch)

        test_inputs = dataset.inputs[test_idx]
        test_targets = dataset.targets[test_idx]

        metrics_to_values = model.test(test_inputs, test_targets)

        for metric, value in metrics_to_values.items():
            if metric not in metrics_to_avg_values:
                metrics_to_avg_values[metric] = 0

            metrics_to_avg_values[metric] += value/splits

        fold += 1

    print("\n########################################")
    print("Cross validation with %d folds complete." % splits)
    for metric, value in metrics_to_avg_values.items():
        print('Average {0}: {1:10.4f}'.format(metric, value))
