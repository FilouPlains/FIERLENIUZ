"""Program used to transform a whole `.out` pyHCA file into a corpus.
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "25/04/2023"
__version__ = "1.1.0"
__copyright__ = "CC BY-SA"

#  [N]
import numpy as np

# [T]
from tqdm import tqdm


def to_peitsch_code(hc: list) -> int:
    """Translate a hydrophobic cluster into a Peitsch code.

    Parameters
    ----------
    code : np.array[bool]
        The hydrophobic cluster.

    Returns
    -------
    int
        The Peitsch code.
    """
    hc = np.array(hc) == "1"

    # A vector of power from 0 to n by 1.
    power_vec: object = np.arange(hc.shape[0] - 1, -1, -1)[hc]
    # A vector of 2.
    two_vec: object = np.full(hc.shape[0], 2)[hc]
    power: object = np.power(two_vec, power_vec)

    return np.sum(power)


def parse_hca_file(path: str) -> object:
    """Parse a pyHCA `.out` file and output a corpus and other data.

    Parameters
    ----------
    file : `str`
        The `.out` path.

    Returns
    -------
    `tuple`
        A `list` with parse data and `np.ndarray` to have the Peitsch code/HC
        correspondance.
    """
    corpus: list = []
    sentence: list = []
    hc_to_peitch = {}
    peitch_to_hc = []

    # Read the file.
    with open(path, "r", encoding="utf-8") as file:
        for line in tqdm(list(file), desc="PARSING `.OUT` FILE"):
            # To keep the domain's ID.
            if line[0] == ">":
                if sentence != []:
                    corpus += [sentence]
                    sentence = []

                continue

            if not line.startswith("cluster"):
                continue

            # Transform the line into a list.
            hc: "str" = line.split()[-1]

            # Remove this special cas, because "1" are not HC.
            if hc == "1":
                continue

            if hc not in hc_to_peitch:
                hc_to_peitch[hc] = to_peitsch_code([*hc])
                peitch_to_hc += [[hc, hc_to_peitch[hc]]]

            sentence += [hc_to_peitch[hc]]

    corpus += [sentence]

    return corpus, np.array(peitch_to_hc)


if __name__ == "__main__":
    # Convert the list into a numpy array and print it.
    print(parse_hca_file("data/REDUNDANCY_DATASET/cd-hit_30.out"))
