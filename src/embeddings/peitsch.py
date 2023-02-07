"""This module contains a function to translate hydrophobic cluster into
Peitsch code.
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "07/02/2023"
__version__ = "1.0.2"
__copyright__ = "CC BY-SA"

# [N]
import numpy as np

# [T]
from tqdm import tqdm


class Peitsch:
    """An object to manipulate and add Peitsch code.
    """

    # The size of `peitsch` or how many Peitsch code have been added.
    size: int = 0
    # The position for the `next()` function.
    position: int = -1
    # Hydrophobic cluster, concentration, cluster size, regular secondary
    # structure, representatives numbers.
    characteristic: "dict[str: list]" = {}
    peitsch: "np.array[str]" = np.array([])

    def __init__(self, ssr: "dict[str: str]") -> None:
        """Instantiate Peitsch object with a already parsed HC database with
        regular secondary structure.

        Parameters
        ----------
        ssr : dict[str: str]
            The already parsed HC database.
        """
        self.ssr = ssr

    def __str__(self) -> str:
        """Redefine how to print the object.

        Returns
        -------
        str
            What is going to be printed with `print()`.
        """
        message: str = (f"Peitsch list of size {self.size}. Actual list is:"
                        f"\n {self.peitsch}")

        return message

    def __iter__(self) -> np.array[str]:
        """Return an "iterator" on all extract Peitsch code.

        Returns
        -------
        np.array[str]
            An numpy array to iterate on with all Peitsch code.
        """
        return self.peitsch

    def __next__(self) -> str:
        """Redefine the `next()` comportement.

        Returns
        -------
        str
            A Peitsch code.

        Raises
        ------
        StopIteration
            When you try to access to an element out of the list.
        """
        if -1 <= self.position < self.size:
            self.position += 1

            return self.peitsch[self.position]
        else:
            raise StopIteration

    def reset_position(self) -> None:
        """Set the position to `-1`. Act like a reset for `next()` function.
        """
        self.position = -1

    def __add__(self, code: "np.array[bool]") -> list:
        """Add a (new) hydrophobic cluster to the object.

        Parameters
        ----------
        code : np.array[bool]
            A hydrophobic cluster.

        Returns
        -------
        list
            A list of characteristics link to the added hydrophobic cluster.
        """
        peitsch: str = self.translate(code)
        concentration: int = self.concentration(code)
        size: int = code.shape[0]

        if peitsch in self.srr:
            ssr: str = self.ssr[peitsch]
        else:
            ssr: str = "N"

        if peitsch in self.characteristic:
            representative: int = self.characteristic[peitsch][4] + 1
        else:
            representative: int = 1

        self.characteristic[peitsch] = [code, concentration, size, ssr,
                                        representative]
        self.peitsch = np.append(self.peitsch, code)
        self.size += 1

        return self.characteristic[peitsch]

    def __iadd__(self, code) -> None:
        """Add a (new) hydrophobic cluster to the object.

        Parameters
        ----------
        code : np.array[bool]
            A hydrophobic cluster.
        """
        peitsch: str = self.translate(code)
        concentration: int = self.concentration(code)
        size: int = code.shape[0]

        if peitsch in self.srr:
            ssr: str = self.ssr[peitsch]
        else:
            ssr: str = "N"

        if peitsch in self.characteristic:
            representative: int = self.characteristic[peitsch][4] + 1
        else:
            representative: int = 1

        self.characteristic[peitsch] = [code, concentration, size, ssr,
                                        representative]
        self.peitsch = np.append(self.peitsch, code)
        self.size += 1

    def concentration(self, code: "np.array[bool]") -> int:
        """Compute the concentration of strong hydrophobic residues in a given
        cluster.

        Parameters
        ----------
        code : np.array[bool]
            The hydrophobic cluster.

        Returns
        -------
        int
            The concentration.
        """
        score: int = 0
        shift: int = 0

        for cluster in code:
            # Increase the score and the value to increase it. So actually,
            # the score evolve like so: [1, 3, 6, 10, 15, 21, ...].
            if cluster:
                shift += 1
                score += shift
            else:
                shift = 0

        return score

    def translate(self, code: "np.array[bool]") -> int:
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
        # A vector of power from 0 to n by 1.
        power_vec: object = np.arange(0, code.shape[0])[code]
        # A vector of 2.
        two_vec: object = np.full(code.shape[0], 2)[code]
        power: object = np.power(two_vec, power_vec)

        return np.sum(power)
