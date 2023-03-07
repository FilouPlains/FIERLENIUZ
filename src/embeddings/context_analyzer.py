"""Compute the divergence between to list, based on `An O(NP) Sequence
Comparison Algorithm` by Sun WU, Udi MANBER, and Gene MYERS in Information
Processing Letters, or in the next link:
https://publications.mpi-cbg.de/Wu_1990_6334.pdf
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "07/03/2023"
__version__ = "1.1.0"
__copyright__ = "CC BY-SA"

# [N]
import numpy as np


class ContextAnalyzer:
    """To create an object to align to set of objects. 
    """

    def __init__(
        self,
        seq_a: "list | str | np.array",
        seq_b: "list | str | np.array"
    ) -> None:
        """Instantiate an `ContextAnalyzer` object.

        Parameters
        ----------
        seq_a : list | str | np.array
            The first sequence-context to compare.
        seq_b : list | str | np.array
            The second sequence-context to compare.
        """
        # Gets sequence-context length.
        self.len_a: int = len(seq_a)
        self.len_b: int = len(seq_b)

        # Invert sequence-context and length so `seq_a` is always the longest
        # one.
        if self.len_b > self.len_a:
            self.len_b, self.len_a = self.len_a, self.len_b
            seq_b, seq_a = seq_a, seq_b

        self.seq_a: str = seq_a
        self.seq_b: str = seq_b

        print(self.seq_a)
        print(self.seq_b)

        # Create an array full of `-1`, of seq_a and seq_b length + 3.
        self.array: object = np.full((self.len_a + self.len_b + 3), -1)
        self.distance: "list[int]" = [None, None]

    def __str__(self) -> str:
        """Change the `print()` function message.

        PRIVATE
        -------

        Returns
        -------
        str
            `print()` the distance if it is already compute. Else, `print()` a
            message to say that the distance is not computed. Do it for all
            implemented methods.
        """
        algo_name: "list[str]" = ["O(NP) sequence algorithm",
                                  "Bray-Curtis formula"]
        to_print: str = ""

        # To check if the distance is compute.
        for i, distance in enumerate(self.distance):
            if distance is not None:
                to_print += (f"- Distance compute with {algo_name[i]} is equal "
                             f"to \"{distance:.1f}\".\n")
            else:
                to_print += (f"- Distance compute with {algo_name[i]} was not "
                             "compute.\n")

        return to_print

    def onp_sequence_comparison(self) -> None:
        """The next implemented algorithm is available in `An O(NP) Sequence
        Comparison Algorithm` by Sun WU, Udi MANBER, and Gene MYERS in 
        Information Processing Letters, or in the next link:
        https://publications.mpi-cbg.de/Wu_1990_6334.pdf

        Based on two given `list | np.array | str`, the algorithm computes a the
        difference between those two. It is one the fastest algorithm to do
        this.
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
                self.array[i + shift] = self.__snake(
                    i,
                    max(self.array[i - 1 + shift] + 1,
                        self.array[i + 1 + shift])
                )

            for i in range(delta + p, delta, -1):
                self.array[i + shift] = self.__snake(
                    i,
                    max(self.array[i - 1 + shift] + 1,
                        self.array[i + 1 + shift])
                )

            self.array[delta + shift] = self.__snake(
                delta,
                max(self.array[delta - 1 + shift] + 1,
                    self.array[delta + 1 + shift])
            )

        # Sequence difference and double of p.
        self.distance[0] = delta + 2 * p

    def __snake(self, i: int, y: int) -> int:
        """Snake function to follow diagonal edges.

        PRIVATE
        -------

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

    def bray_curtis(self):
        """Bray-Curtis distance, which is 1 - Sørensen-Dice coefficient,
        following the next formula:

        ```
                        2 *|A ∩ B|
        distance = 1 - ————————————
                         |A| + |B|
        ```

        In other word, we compute the distance between to finished `np.array`.
        Note that `|A|` signify the length of a given (computed) set.
        """
        # To transform a string in a `np.array`.
        if isinstance(self.seq_a, str):
            seq_a: np.array = np.array([*self.seq_a])
        else:
            seq_a: np.array = np.array(self.seq_a)

        # To transform a string in a `np.array`.
        if isinstance(self.seq_b, str):
            seq_b: np.array = np.array([*self.seq_b])
        else:
            seq_b: np.array = np.array(self.seq_b)

        # Compute the intersection.
        intersection: int = np.intersect1d(seq_b, seq_a).shape[0]
        # Compute the coefficient.
        coef: float = intersection / (seq_a.shape[0] + seq_b.shape[0])

        # Add the coefficient to the list.
        self.distance[1] = 1 - coef


if __name__ == "__main__":
    Context: object = ContextAnalyzer("chienn", "niche")
    Context.onp_sequence_comparison()
    Context.bray_curtis()
    print(Context)

    Context: object = ContextAnalyzer("abc", "abd")
    Context.onp_sequence_comparison()
    Context.bray_curtis()
    print(Context)

    Context: object = ContextAnalyzer([201, 147, 21, 45, 307],
                                      [201, 147, 25, 49, 307])
    Context.onp_sequence_comparison()
    Context.bray_curtis()
    print(Context)

    Context: object = ContextAnalyzer("abcdefg", "hijklmnop")
    Context.onp_sequence_comparison()
    Context.bray_curtis()
    print(Context)
