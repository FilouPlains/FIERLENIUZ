"""This module is used to transform a `.fasta` file to a pyHCA`.out` format.
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "19/04/2023"
__version__ = "1.0.1"
__copyright__ = "CC BY-SA"

# [N]
import numpy as np

# [T]
from textwrap import fill
from tqdm import tqdm


class Domain_Manipulator:
    """To manipulate domains.
    """
    # Residues considered as "strong hydrophobic".
    HYDROPHOBIC: "list[str]" = ["V", "I", "L", "F", "M", "Y", "W"]
    # Number of residues separating two clusters.
    SIZE_SEPARATOR: int = 4
    # One of those residues will systematically separate two clusters.
    RESIDUE_SEPARATOR: "list[str]" = ["P"]

    def __init__(self, name: str):
        """Instantiate a new `Domain_Manipulator` object.

        Parameters
        ----------
        name : `str`
            The domain name.
        """
        # Domain name.
        self.name: str = name
        # Domain size.
        self.size: int = 0
        # Domain sequence.
        self.sequence: str = ""
        # Domain transform in a list of hydrophobic clusters.
        self.hca: "list[str]" = []

    def __iadd__(self, sequence: str) -> object:
        """Add a sequence part to this domain.

        Parameters
        ----------
        sequence : `str`
            The sequence to append to the domain.

        Returns
        -------
        object
            `self`.
        """
        # Add the sequence in UPPER CASE.
        self.sequence += sequence.upper().strip()
        # Increase domain size.
        self.size += len(sequence)

        return self

    def __add__(self, sequence: str) -> object:
        """Add a sequence part to this domain.

        Parameters
        ----------
        sequence : `str`
            The sequence to append to the domain.

        Returns
        -------
        object
            `self`.
        """
        # Add the sequence in UPPER CASE.
        self.sequence += sequence.upper().strip()
        # Increase domain size.
        self.size += len(sequence)

        return self

    def __str__(self) -> str:
        """Print a personalize message when using `print()` function.

        Returns
        -------
        `str`
            The personalize message.
        """
        # Format the sequence so there is a line break each 80 characters.
        sequence: str = fill(self.sequence, width=80)
        to_print: str = f">{self.name} size={self.size}\n{sequence}\n"

        # Add HC informations if they were computed.
        if self.hca != []:
            hca: str = fill(" ".join(self.hca), width=80)
            to_print += f">HCA_sequence\n{hca}\n"

        return to_print

    def to_hca(self) -> None:
        """Convert this domain sequence into a hydrophobic clusters sequence.
        """
        # Set the separators.
        sep: str = "0" * self.SIZE_SEPARATOR
        # Transform a string to a list of characters.
        sequence: "list[str]" = [*self.sequence]

        # Each "strong hydrophobic" residues are convert to `True` and the rest
        # to `False`. After, we convert those boolean values to `int` (1 and 0)
        # and then to `str`.
        hca: np.ndarray = np.isin(sequence,
                                  self.HYDROPHOBIC).astype(int).astype(str)
        # When we meet a residue that systematically separated two clusters, we
        # replaced it with `sep`.
        hca[np.isin(sequence, self.RESIDUE_SEPARATOR)] = sep

        # We only used string quick operations. We join the `hca` `np.ndarray`
        # into a single string that we split each `sep`. Then with `filter()`,
        # we throw away every empty `str`. After that, we strip all `0` form
        # the formed clusters. To finish, we convert all of this into a
        # `list()`.
        self.hca = list(map(
            lambda hc_i: hc_i.strip("0"),
            filter(None, "".join(hca).split(sep))
        ))

    def to_pyhca_format(self) -> str:
        """Return a string representation of the hydrophobic clusters into a
        pyHCA`.out` format.

        Returns
        -------
        `str`
            The string representation of the hydrophobic clusters.
        """
        # First line with domains.
        format_str: str = f">{self.name} {self.size} nan nan\n"
        # Second line with foldable segments.
        format_str += f"domain\t1\t{self.size}\tnan\tnan\n"
        # We append this because join do not put elements between a `list`.
        format_str += "cluster\t1\t2\t"
        # Join a `list` into a `str`.
        format_str += "\ncluster\t1\t2\t".join(self.hca) + "\n"

        return format_str


def hca_extractor(fasta: str, out_path: str, size_limit: int = 1):
    """Extract all hydrophobic clusters from `.fasta` file and write a
    pyHCA`.out` file.

    Parameters
    ----------
    fasta : `str`
        The path to the `.fasta` file.
    out_path : `str`
        The path where to write the pyHCA`.out` file.
    size_limit : `int`, optional
        If a amino sequence is strictly inferior to the give value, do not
        write it into the file. By default `1`.
    """
    # File's header.
    HEADER: str = ("# HCA extraction\n# \n# >{domain_name} {domain_size} nan "
                   "nan\n# domain	1	{domain_size}	nan	nan\n# cluster"
                   "	1	2	{HC}\n#\n# Cluster separator:"
                   " [PROLINE] or [4 RESIDUES]\n#\n\n")

    # Read the `.fasta` file and write the pyHCA`.out` file.
    with open(fasta, "r", encoding="utf-8") as in_file, \
            open(out_path, "w", encoding="utf-8") as out_file:
        out_file.write(HEADER)
        domain: Domain_Manipulator = None

        # Parsing file's lines.
        for line in tqdm(list(in_file), "PARSING `.fasta` FILE"):
            split_line: "list[str]" = line.split()

            # If we have the domain line.
            if line[0] == ">":
                # When `domain` is instantiated and we have a correct sequence
                # size.
                if domain != None and domain.size >= size_limit:
                    # Convert the residue sequence to a hydrophobic clusters
                    # sequence.
                    domain.to_hca()
                    # Write lines into the file.
                    out_file.write(domain.to_pyhca_format())

                # Instantiate a new `Domain_Manipulator` object.
                domain = Domain_Manipulator(name=split_line[0][1:])
                continue

            # Append the sequence line.
            domain += line.strip()


if __name__ == "__main__":
    hca_extractor(
        fasta="/home/lrouaud/Téléchargements/cd-hit/cd-hit_70.fasta",
        out_path="/home/lrouaud/Téléchargements/cd-hit/cd-hit_70.out",
        size_limit=30
    )
