"""
Calculate common metrics such as accuracy.
"""


def correct_predictions_two_class(outputs, targets):
    """
    Calculates the number of correct predictions made by the network.

    Use this if the network outputs a simple probability for a two-class 
    classification problem.

    @param outputs: the outputs of the network as a torch.Tensor
    @param targets: what you want the network to output, as a torch.Tensor
    """

    return ((outputs > 0.5) == targets.byte()).sum().item()
