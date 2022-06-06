import collections
import numpy as np

import logging

logger = logging.getLogger(__name__)


def relative(current, prev, eps=1e-6):
    return (np.abs(current - prev) + eps) / (np.abs(prev) + eps)


def absolute(current, prev):
    return np.abs(current - prev)


_diff = dict(relative=relative, absolute=absolute)


def flatten_shared(shared_list):
    return np.concatenate([sh.flatten() for sh in shared_list])


def elbos_callback(elbos, window=50, every=50, threshold=1e-2):
    """
    Computes mean of history of elbos and checks if the relative change is smaller the threshold
    """
    i = len(elbos) - 1
    if i % every or i <= 2 * window:
        return

    prev_avg = np.median(elbos[-2 * window : -window])
    latest_avg = np.median(elbos[-window:])
    error = _diff["relative"](latest_avg, prev_avg)
    if error < threshold:
        logger.debug(f"Convergence achieved at {i}")
        raise StopIteration


def vi_callback_params(
    i, new_params, old_params, every=100, tolerance=1e-3, diff="relative", ord=np.inf
):
    """
    Returns the now old params. Stops iterable if convergence achieved.
    """
    if i % every or i < every:
        return
    if old_params is None:
        return flatten_shared(new_params)
    new_params = flatten_shared(new_params)
    delta = _diff[diff](new_params, old_params)  # type: np.ndarray
    norm = np.linalg.norm(delta, ord)
    if norm < tolerance:
        logger.debug(f"Convergence achieved at {i}")
        raise StopIteration
    return new_params


def tree_callback(
    i, new_params, old_params, every=100, tolerance=1e-3, diff="relative", ord=np.inf
):
    """
    Returns the now old params. Stops iterable if convergence achieved.
    """
    raise NotImplementedError
