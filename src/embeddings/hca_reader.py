"""Program used to transform the whole `.hca` file into a numpy array.
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "31/01/2023"
__version__ = "1.0.2"
__copyright__ = "CC BY-SA"

#  [N]
import numpy as np

# [T]
from tqdm import tqdm


def parse_hca_file(path: str) -> object:
    """Parse a pyHCA `.out` file and output a numpy array.

    Parameters
    ----------
    file : str
        The `.out` path.

    Returns
    -------
    object
        A numpy array to save.
    """
    hca_out: object = []

    # Read the file.
    with open(path, "r", encoding="utf-8") as file:
        domain_name: str = ""

        for line in tqdm(file, desc="PARSING FILE"):
            # To keep the domain's ID.
            if line.startswith(">"):
                domain_name = line.split()[0][1:]
                segment: "list[list[int]]" = []

                continue
            # Skip all first comments or domain lines.
            elif not domain_name:
                continue
            elif line.startswith("domain"):
                segment += [[int(line.split()[1]), int(line.split()[2])]]

            # Transform the line into a list.
            split_line: "list[str]" = line.split()
            segment_id: int = 0

            for i, pos in enumerate(segment):
                # The cluster interval is not included in the segment interval.
                if not (pos[0] <= int(split_line[1]) <= pos[1] and
                        pos[0] <= int(split_line[2]) <= pos[1]):
                    continue

                # Increasing the segment ID.
                segment_id = i + 1
                break

            # Actual cluster not in a foldable segment.
            if segment_id <= 0:
                continue

            # Keep the domain's ID and a numpy array as a bool type.
            read_in: "list[str]" = [domain_name, segment_id,
                                    np.array([*split_line[-1]]) == "1"]

            hca_out += [read_in]

    return np.asarray(hca_out, dtype=object)


if __name__ == "__main__":
    # File emplacement.
    path: str = "data/hca.out"

    # Convert the list into a numpy array and print it.
    print(parse_hca_file(path))
