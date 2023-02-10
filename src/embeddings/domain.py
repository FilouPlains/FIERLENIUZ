"""This module contains an object to manipulate domain information.
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "07/02/2023"
__version__ = "1.0.3"
__copyright__ = "CC BY-SA"

# [N]
import numpy as np
# [S]
import sys


class Domain:
    """An object to manipulate and add domain informations.
    """
    # The size of `peitsch` or how many Peitsch code have been added.
    size: int = 0
    # Hydrophobic cluster, concentration, cluster size, regular secondary
    # structure, representatives numbers.
    domain: "dict[str: list]" = {}

    def __init__(self) -> None:
        """Instantiate Domain object.
        """
        pass

    def __str__(self) -> str:
        """Redefine how to print the object.

        Returns
        -------
        str
            What is going to be printed with `print()`.
        """
        to_print: str = (f"Domain dictionary of size {self.size}. Actual "
                         f"dictionary is:\n {self.peitsch}")

        return to_print

    def __add__(self, characteristic: "list[str, int, float]") -> object:
        """Add a (new) domain to the object.

        Parameters
        ----------
        characteristic : list[str, int, float]
            A hydrophobic cluster.

        Returns
        -------
        object
            self.
        """
        if len(characteristic) != 3:
            sys.exit("[Err## 10]")
                
        # Add a Peitsch code.
        peitsch: str = self.translate(code)
        # Add a concentration value.
        concentration: int = self.concentration(code)
        # Add a size value.
        size: int = code.shape[0]

        # Add a regular secondary structure.
        if peitsch in self.ssr:
            ssr: str = self.ssr[peitsch]
        # In case of not corresponding SSR found.
        else:
            ssr: str = "N"

        # Number of Peitsch code found in a "corpus".
        if peitsch in self.characteristic:
            representative: int = self.characteristic[peitsch][4] + 1
        else:
            representative: int = 1

        self.characteristic[peitsch] = [code, concentration, size, ssr,
                                        representative]
        self.peitsch = np.append(self.peitsch, peitsch)
        self.size += 1

        return self

    def __iadd__(self, code) -> None:
        """Add a (new) hydrophobic cluster to the object.

        Parameters
        ----------
        code : np.array[bool]
            A hydrophobic cluster.
        """
        # Add a Peitsch code.
        peitsch: str = self.translate(code)
        # Add a concentration value.
        concentration: int = self.concentration(code)
        # Add a size value.
        size: int = code.shape[0]

        # Add a regular secondary structure.
        if peitsch in self.ssr:
            ssr: str = self.ssr[peitsch]
        # In case of not corresponding SSR found.
        else:
            ssr: str = "N"

        # Number of Peitsch code found in a "corpus".
        if peitsch in self.characteristic:
            representative: int = self.characteristic[peitsch][4] + 1
        else:
            representative: int = 1

        self.characteristic[peitsch] = [code, concentration, size, ssr,
                                        representative]
        self.peitsch = np.append(self.peitsch, peitsch)
        self.size += 1

        return self