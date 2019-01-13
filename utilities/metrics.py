"""
Calculate common metrics such as accuracy.
"""


def accuracy(outputs, targets):
    """
    Calculates the accuracy of some predictions.

    Use this if the network outputs a simple probability for a two-class 
    classification problem.

    @param outputs: the outputs of the network as a torch.Tensor
    @param targets: what you want the network to output, as a torch.Tensor
    @return: the accuracy, a single number
    """

    outputs_len = outputs.size()[0]
    return ((outputs > 0.5) == targets.byte()).sum().item()/outputs_len


def precision(outputs, targets):
    """
    Calculates precision, defined as True Positives/(True Positives + False Negatives).

    @param outputs: the outputs of the network as a torch.Tensor
    @param targets: what you want the network to output, as a torch.Tensor
    @return: the precision, a single number
    """

    true_positives = (outputs > 0.5)*(targets == 1).sum().item()
    false_positives = (outputs > 0.5)*(targets == 0).sum().item()

    return true_positives/(true_positives + false_positives)
