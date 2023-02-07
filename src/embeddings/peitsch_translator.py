"""This module contains a function to translate hydrophobic cluster into
Peitsch code.
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "03/02/2023"
__version__ = "1.0.2"
__copyright__ = "CC BY-SA"

# [N]
import numpy as np

# [T]
from tqdm import tqdm


def peitsch_translator(data: object, max_size: int = 100) -> "list[list[str]]":
    """This function is used to transform input hydrophobic data into Peitsch
    code.

    Parameters
    ----------
    data : object
        A `hca.npy` file, containing ID and hydrophobic clusters. This is those
        last one that are going to be translated into Peitsch code.

    Returns
    -------
    list[list[str]]
        A list of list (= sentence) of Peitsch code.
    """
    # To contain all Peitsch code.
    peitsch_list: list = []
    sentence: list = []

    # A vector of power from 0 to n by 1.
    power_vec: object = np.arange(0, max_size)
    # A vector of 2.
    two_vec: object = np.full(max_size, 2)
    power: object = np.power(two_vec, power_vec)

    shift: str = data[0][0].split("_")[0]
    
    print(np.unique(np.transpose(data)[1]))

    for line in tqdm(data, desc="TRANSLATING CODES"):
        # Get the length of the hydrophobic cluster.
        length: int = line[1].shape[0]

        # Take the vector of 2 to the power of the power vector, then sum
        # the whole vector.
        peitsch_code: str = np.sum(power[0:length][line[1]])

        # Get domain's ID.
        identifier: str = line[0].split("_")[0]

        # If the domain not change, keep being "in the same sentence".
        if shift == identifier:
            sentence += [peitsch_code]
        # The domaine change, adding the sentence to the list. Resetting the
        # sentence.
        else:
            # Adding the Peitsch code to the list.
            peitsch_list += [sentence]

            sentence = [peitsch_code]
            shift = identifier

    # Adding last sentence to the corpus.
    peitsch_list += [sentence]

    return peitsch_list
