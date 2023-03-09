"""Generates a `.csv` file from a `.npy` file.
"""


__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "09/023/2023"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


# [N]
import numpy as np

# [O]
from os import popen


def cluster_to_str(cluster_line: np.array) -> str:
    """Transforms a numpy boolean array to a string.

    Parameters
    ----------
    cluster_line : np.array
        The numpy array to be transformed.

    Returns
    -------
    str
        The hydrophobic cluster in string format.
    """
    # Convert the numpy to int then to string so join can work properly.
    return "".join(cluster_line.astype(int).astype(str))


def extract_array(path: str) -> np.array:
    """Manipulation to gets a numpy array from a file containing a `*`.

    Parameters
    ----------
    path : str
        The path where the data have to be extract.

    Returns
    -------
    np.array
        The extracted array.
    """
    true_path: str = popen(f"ls {path}")

    array: object = np.transpose(np.load(true_path.readlines()[0].strip(),
                                         allow_pickle=True))

    return array


if __name__ == "__main__":
    # Data extraction.
    characteristic = extract_array("data/peitsch2vec/default_domain/"
                                   "characteristics*.npy")
    embedding_domain = extract_array("data/peitsch2vec/default_domain/"
                                     "embedding*.npy")
    embedding_segment = extract_array("data/peitsch2vec/default_segments/"
                                      "embedding*.npy")

    # This part is done so we explicit what is the meaning of each extracted
    # data.
    code_peitsch: np.array = characteristic[0]
    # Convert all np.array of bool to str hydrophobic clusters.
    cluster: np.array = np.array(list(map(cluster_to_str, characteristic[1])))
    concentration: np.array = characteristic[2]
    size: np.array = characteristic[3]
    ssr: np.array = characteristic[4]
    count: np.array = characteristic[5]
    # Extract the vectors' norm.
    norm_domain: np.array = np.linalg.norm(embedding_domain, axis=0)
    norm_segment: np.array = np.linalg.norm(embedding_segment, axis=0)

    # Concatenante all data in a matrix and transpose it.
    data_matrix: np.array = np.transpose(np.array([
        code_peitsch,
        cluster,
        concentration,
        size,
        ssr,
        count,
        norm_domain.round(4),
        norm_segment.round(4)
    ]))

    # Quickest way to save the whole matrix into a `.csv` format.
    np.savetxt(
        "data/characteristic.csv",
        data_matrix,
        fmt="%s",
        delimiter=",",
        header=("CODE_PEITSCH,CLUSTER,HYDROPHOBIC_SCORE,SIZE,SSR,COUNT,"
                "NORM_DOMAIN,NORM_SEGMENT"),
        comments=""
    )
