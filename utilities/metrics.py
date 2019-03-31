"""
Calculate common metrics such as accuracy.
"""


def accuracy(outputs, targets):
    """
    Calculates the accuracy of some predictions. Inputs are one hot encoded.

    @param outputs: the outputs of the network as a torch.Tensor
    @param targets: what you want the network to output, as a torch.Tensor
    @return: the accuracy, a single number
    """

    length = outputs.size()[0]
    classes = len(outputs[0])
    predictions = torch.argmax(outputs, 1)
    targets_single_number = torch.argmax(targets, 1)

    return (predictions == targets_single_number).sum().item()/length


def precision(outputs, targets):
    """
    Calculates precision, defined as True Positives/(True Positives + False Negatives).

    @param outputs: the outputs of the network as a torch.Tensor
    @param targets: what you want the network to output, as a torch.Tensor
    @return: the precision, a single number
    """

    true_positives = ((outputs > 0.5)*(targets == 1)).sum().item()
    false_positives = ((outputs > 0.5)*(targets == 0)).sum().item()

    if true_positives + false_positives == 0:
        return 0

    return true_positives/(true_positives + false_positives)
