"""This module contains a function to translate hydrophobic cluster into
Peitsch code.
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "25/01/2023"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# [N]
import numpy as np

# [T]
from tqdm import tqdm


def peitsch_translator(data: object) -> object:
    """This function is used to transform input hydrophobic data into Peitsch
    code.

    Parameters
    ----------
    data : object
        A `hca.npy` file, containing ID and hydrophobic clusters. This is those
        last one that are going to be translated into Peitsch code.

    Returns
    -------
    object
        A numpy array of Peitsch code.
    """
    # To contain all Peitsch code.
    peitsch_list: list = []

    for line in tqdm(data, desc="PARSING FILE"):
        # Get the length of the hydrophobic cluster.
        length: int = line[1].shape[0]

        # A vector of power from 0 to n by 1.
        power_vec: object = np.arange(0, length)
        # A vector of 2.
        two_vec: object = np.full(length, 2)
        # Take the vector of 2 to the power of the power vector, then sum
        # the whole vector.
        peitsch_code: int = np.sum(np.power(two_vec, power_vec)[line[1]])

        # Adding the Peitsch code to the list.
        peitsch_list += [peitsch_code]

    return np.array(peitsch_list, dtype=np.int32)


if __name__ == "__main__":
    peitsch: object = peitsch_translator(np.load("embeddings/hca.npy",
                                                 allow_pickle=True))

    print(peitsch)
