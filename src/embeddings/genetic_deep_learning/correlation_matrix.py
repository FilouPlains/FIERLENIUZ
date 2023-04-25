"""To create an object to compute correlation between two matrix.
"""


__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "25/04/2023"
__version__ = "1.1.0"
__copyright__ = "CC BY-SA"

# [N]
import numpy as np

# [S]
from sys import path
# [T]
from tqdm import tqdm

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
        self.matrix: np.ndarray = np.zeros((length, length), dtype=int)

        # Parse through the half-matrix.
        for i, hc_i in enumerate(tqdm(self.hc_list[:-1],
                                      desc="    CREATING MATRIX")):
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

    def __str__(self):
        """Change the `print()` function comportment.

        Returns
        -------
        `str`
            The new `print()` message, which is the matrix of distance edition.
        """
        to_print: str = "┏"
        # Get the matrix's cell size.
        max_line: np.ndarray = [0] + list(self.matrix.max(axis=0))
        # Line with number/letter.
        line: str = "┃"
        # Line that separates two number/letter lines.
        h_line: str = "┣"
        # First header line.
        bold_h_line: str = "┣"
        # Last bottom line.
        b_line: str = "┗"

        longest_hc: str = ""

        # Get the maximum size for the first row.
        for hc_i in self.hc_list:
            if len(hc_i) > len(longest_hc):
                longest_hc = hc_i

        # Initiate matrix creation with header.
        for i, hc_i in enumerate([longest_hc] + list(self.hc_list)):
            max_line[i] = max(len(f"{max_line[i]}"), len(hc_i))

            to_print += "━" * (max_line[i] + 2) + "┳"

            # To take in case the first column.
            if i != 0:
                h_line += "─" * (max_line[i] + 2) + "┼"
                b_line += "─" * (max_line[i] + 2) + "┴"
                bold_h_line += "━" * (max_line[i] + 2) + "╇"
                line += f"{hc_i:^{max_line[i] + 2}}┃"
            else:
                h_line += "━" * (max_line[i] + 2) + "╉"
                b_line += "━" * (max_line[i] + 2) + "┹"
                bold_h_line += "━" * (max_line[i] + 2) + "╋"
                line += " " * (max_line[i] + 2) + "┃"

        # Change the last character to close the matrix.
        to_print = to_print[:-1] + "┓\n"
        h_line = h_line[:-1] + "┤\n"
        bold_h_line = bold_h_line[:-1] + "┩\n"
        line = line[:-1] + "┃\n"
        b_line = b_line[:-1] + "┘\n"

        # Append to the final `print()` message.
        to_print += line + bold_h_line

        # Fill the matrix.
        for i, row in enumerate(self.matrix):
            line = "┃"

            for j, cell in enumerate([self.hc_list[i]] + list(row)):
                if j != 0:
                    line += f"{cell:^{max_line[j] + 2}}│"
                else:
                    line += f" {cell:>{max_line[j]}} ┃"

            line = line[:-1] + "│\n"

            # To take in case the first column.
            if i + 1 != self.matrix.shape[1]:
                to_print += line + h_line
            else:
                to_print += line + b_line

        # Do no return the last "\n" when printing the matrix, as far as the
        # `print()` function already do it.
        return to_print[:-1]

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
        w2v_hc_sort: np.ndarray = np.argsort(w2v_hc)
        hc_sort: np.ndarray = np.argsort(self.hc_list)

        w2v_hc_index: np.ndarray = w2v_hc_sort[np.isin(w2v_hc[w2v_hc_sort],
                                                       self.hc_list)]
        hc_list_index: np.ndarray = hc_sort[np.isin(self.hc_list[hc_sort],
                                                    w2v_hc)]

        # To reindex the bigger matrix, because they could be missing Peitsch
        # code after w2v's vector generation.
        dist_mat: np.ndarray = self.matrix[hc_list_index, :][:, hc_list_index]
        cos_mat: np.ndarray = cosine_matrix[w2v_hc_index, :][:, w2v_hc_index]

        # To check if the size is sufficient or not.
        size: int = dist_mat[np.triu_indices(dist_mat.shape[0], k=1)].shape[0]

        # Compute a correlation coefficient between two half flatten matrix,
        # without the diagonal.
        if size > 1:
            coefficient: float = np.corrcoef(
                dist_mat[np.triu_indices(dist_mat.shape[0], k=1)],
                cos_mat[np.triu_indices(cos_mat.shape[0], k=1)]
            )[0][1]
        else:
            coefficient: float = 0

        # Return R².
        return coefficient ** 2


if __name__ == "__main__":
    cor_mat: list = [[0, 3, 1],
                     [3, 0, 2],
                     [1, 2, 0]]

    c_c: ComputeCorrelation = ComputeCorrelation(["111", "101", "1001"])
    print(c_c)

    print(c_c.compute_correlation(np.array(cor_mat),
                                  np.array(["1001", "111", "101"])))

    print(c_c.compute_correlation(np.array([[1, 2, 3]] * 3),
                                  np.array(["c", "ab", "b"])))
