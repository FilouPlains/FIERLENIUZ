"""Compute the divergence between to list, based on `An O(NP) Sequence
Comparison Algorithm` by Sun WU, Udi MANBER, and Gene MYERS in Information
Processing Letters, or in the next link:
https://publications.mpi-cbg.de/Wu_1990_6334.pdf
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "06/03/2023"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# [N]
import numpy as np


class Alignment:
    """To create an object to align to set of objects. The algorithm is
    available in `An O(NP) Sequence Comparison Algorithm` by Sun WU, Udi MANBER,
    and Gene MYERS in Information Processing Letters, or in the next link:
    https://publications.mpi-cbg.de/Wu_1990_6334.pdf
    """

    def __init__(self, seq_a: str, seq_b: str) -> None:
        """Instantiate an `Alignment` object.

        Parameters
        ----------
        seq_a : str
            The first sequence-context to compare.
        seq_b : str
            The second sequence-context to compare.
        """
        # Gets sequence-context length.
        self.len_a: int = len(seq_a)
        self.len_b: int = len(seq_b)

        # Invert sequence-context and length so `seq_a` is always the longest
        # one.
        if self.len_b > self.len_a:
            self.len_b, self.len_a = self.len_b, self.len_a
            seq_b, seq_a = seq_a, seq_b

        self.seq_a: str = seq_a
        self.seq_b: str = seq_b

        # Create an array full of `-1`, of seq_a and seq_b length + 3.
        self.array: object = np.full((self.len_a + self.len_b + 3), -1)
        self.score: int = None

    def __str__(self) -> str:
        """Change the `print()` function message.

        Returns
        -------
        str
            `print()` the score if it is already compute. Else, `print()` a
            message to say that the score is not computed.
        """
        # To check if the score is compute.
        if self.score:
            return f"Score is equal to {self.score}."

        return "Score note compute."

    def compare(self) -> None:
        """Compute a score to check how much two sequence-context are different.
        """
        p: int = -1
        # Size different between seq_a and seq_b
        delta: int = self.len_a - self.len_b
        # To shift, because in the original paper, they have a negative indexed
        # array.
        shift: int = self.len_a + 1

        # Check when all operation are done and correct.
        while self.array[delta + shift] < self.len_b:
            p += 1

            for i in range(-p, delta, 1):
                self.array[i + shift] = self.snake(
                    i,
                    max(self.array[i - 1 + shift] + 1,
                        self.array[i + 1 + shift])
                )

            for i in range(delta + p, delta, -1):
                self.array[i + shift] = self.snake(
                    i,
                    max(self.array[i - 1 + shift] + 1,
                        self.array[i + 1 + shift])
                )

            self.array[delta + shift] = self.snake(
                delta,
                max(self.array[delta - 1 + shift] + 1,
                    self.array[delta + 1 + shift])
            )

        # Sequence difference and double of p.
        self.score = delta + 2 * p

    def snake(self, i: int, y: int) -> int:
        """Snake function to follow diagonal edges.

        Parameters
        ----------
        i : int
            Array position.
        y : int
            `seq_b` position.

        Returns
        -------
        int
            The furthest distance between to sequence-context.
        """
        x: int = y - i

        while x < self.len_a and y < self.len_b and \
                self.seq_a[x] == self.seq_b[y]:
            x += 1
            y += 1

        return y


if __name__ == "__main__":
    AlignmentMatrix: object = Alignment("chien", "niche")
    AlignmentMatrix.compare()
    print(AlignmentMatrix)

    AlignmentMatrix: object = Alignment("abc", "abd")
    AlignmentMatrix.compare()
    print(AlignmentMatrix)

    AlignmentMatrix: object = Alignment([201, 147, 21, 45, 307],
                                        [201, 147, 25, 49, 307])
    AlignmentMatrix.compare()
    print(AlignmentMatrix)
