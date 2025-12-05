import numpy as np

class BinarySymmetricChannel:
    """
    Simulates a channel that flips bits with probability p.
    """

    def __init__(self, error_probability):
        self.p = error_probability

    def transmit(self, bits):
        """
        Passes bits through the noisy channel.
        """
        noise = np.random.choice([0, 1], size=len(bits), p=[1 - self.p, self.p])
        received_bits = (bits + noise) % 2
        return received_bits, np.sum(noise)