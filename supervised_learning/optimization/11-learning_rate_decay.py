#!/usr/bin/env python3
"""Learning rate decay module (numpy)."""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Update learning rate using stepwise inverse time decay.

    Args:
        alpha: original learning rate.
        decay_rate: weight determining the rate of decay.
        global_step: number of gradient descent passes elapsed.
        decay_step: passes before alpha is decayed again.

    Returns:
        Updated value for alpha.
    """
    return alpha / (1 + decay_rate * np.floor(global_step / decay_step))
