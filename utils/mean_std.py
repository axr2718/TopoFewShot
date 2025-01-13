import numpy as np

def compute_mean_std_err(accuracies: list) -> tuple[float, float]:
    """
    Computes the mean and standard error of a metric list.

    Args:
        metric_list (list): A list containing the metrics of each fold.

    Returns:
        tuple[float, float]: The mean and standard error of the aggregated list.

    """
    mean = np.mean(accuracies)
    std_err = np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))

    return mean, std_err