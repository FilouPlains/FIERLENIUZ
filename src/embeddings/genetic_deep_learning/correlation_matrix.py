"""To create an object to compute correlation between two matrix.
"""


__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "21/04/2023"
__version__ = "1.0.1"
__copyright__ = "CC BY-SA"

# [N]
import numpy as np

# [S]
from sys import path

# Checking parent directory.
path.append("src/embeddings/")
from context_analyzer import PairewiseContextAnalyzer


class ComputeCorrelation:
    """To compute correlation between two matrix.
    """
    def __init__(self, hc_list: np.ndarray):
        """Instantiates an object `ComputeCorrelation`.

        Parameters
        ----------
        hc_list : `np.ndarray`
            A list of HC to use for generating a edition distance matrix.
        """        
        self.hc_list = np.array(hc_list)

        length: int = self.hc_list.shape[0]
        # Generate a matrix of size (len(self.hc_list), len(self.hc_list)).
        self.matrix: np.ndarray = np.zeros((length, length), dtype=float)

        # Parse through the half-matrix.
        for i, hc_i in enumerate(self.hc_list[:-1]):
            for j, hc_j in enumerate(self.hc_list[i + 1:]):
                # Compute edition distance.
                context_analyzer = PairewiseContextAnalyzer(
                    seq_a=hc_i,
                    seq_b=hc_j
                )

                context_analyzer.onp_sequence_comparison(normalize=False)

                # Fill the whole matrix.
                self.matrix[i][i + 1 + j] = context_analyzer.distance[0]
                self.matrix[i + 1 + j][i] = context_analyzer.distance[0]

    def compute_correlation(
        self,
        cosine_matrix: np.ndarray,
        w2v_hc: np.ndarray
    ) -> float:
        """Compute the correlation between two matrix.

        Parameters
        ----------
        cosine_matrix : `np.ndarray`
            The cosine similarity matrix.
        peitch_to_hc : `np.ndarray`
            A list of HC from w2v's vectors.

        Returns
        -------
        `float`
            R²
        """
        # To reindex the bigger matrix, because they could be missing Peitsch
        # code after w2v's vector generation.
        filter: np.ndarray = np.where(np.isin(self.hc_list, w2v_hc))[0]
        dist_mat: np.ndarray = self.matrix[filter, :][:, filter]

        # To check if the size is sufficient or not.
        size: int = dist_mat[np.triu_indices(dist_mat.shape[0], k=1)].shape[0]

        # Compute a correlation coefficient between two half flatten matrix,
        # without the diagonal.
        if size > 1:
            coefficient: float = np.corrcoef(
                dist_mat[np.triu_indices(dist_mat.shape[0], k=1)],
                cosine_matrix[np.triu_indices(cosine_matrix.shape[0], k=1)]
            )[0][1]
        else:
            coefficient: float = 0

        # Return R².
        return coefficient ** 2


if __name__ == "__main__":
    c_c: ComputeCorrelation = ComputeCorrelation(["a", "b", "ab", "c"])
    print(c_c.compute_correlation(np.array([[1, 2]] * 2),
                                  np.array(["c", "ab"])))

    print(c_c.compute_correlation(np.array([[1, 2, 3]] * 3),
                                  np.array(["c", "ab", "b"])))
