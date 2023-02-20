"""Program used to transform the whole `.hca` file into a numpy array.
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "31/01/2023"
__version__ = "1.2.2"
__copyright__ = "CC BY-SA"

#  [N]
import numpy as np

# [T]
from tqdm import tqdm


def parse_hca_file(path: str, by_segment: bool = False) -> object:
    """Parse a pyHCA `.out` file and output a numpy array.

    Parameters
    ----------
    file : str
        The `.out` path.

    Returns
    -------
    object
        A numpy array with parse data.
    """
    hca_out: object = []

    # NB_PEITSCH, NB_FOLDABLE_SEGMENT, SIZE, HCA_SCORE, P-VALUE.
    domain: "dict[str: list[int, float]]" = {}

    # Read the file.
    with open(path, "r", encoding="utf-8") as file:
        domain_name: str = ""

        for line in tqdm(list(file), desc="          PARSING FILE"):
            # Transform the line into a list.
            split_line: "list[str]" = line.split()

            # To keep the domain's ID.
            if line.startswith(">"):
                domain_name = split_line[0][1:]
                segment: "list[list[int]]" = []
                segment_data: "list[list[int, float]]" = []
                
                if not by_segment:
                    domain[domain_name] = [
                        0,
                        int(split_line[1]),
                        float(split_line[2]),
                        float(split_line[3])
                    ]

                continue
            # Skip all first comments or domain lines.
            elif not domain_name:
                continue
            elif line.startswith("domain"):
                if by_segment:
                    if split_line[3] == "nan":
                        score: int = 0
                    else:
                        score: int = float(split_line[3])
                    
                    segment_data += [[
                        0,
                        int(split_line[2]) - int(split_line[1]),
                        score,
                        float(split_line[4])
                    ]]

                segment += [[int(line.split()[1]), int(line.split()[2])]]

                continue

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

            if by_segment:
                seg_name: str = f"{domain_name}_{segment_id}"
                
                if seg_name not in domain:
                    domain[seg_name] = segment_data[segment_id - 1]

                domain[seg_name][0] += 1
            else:
                domain[domain_name][0] += 1

            # Keep the domain's ID and a numpy array as a bool type.
            read_in: "list[str]" = [domain_name, segment_id,
                                    np.array([*split_line[-1]]) == "1"]

            hca_out += [read_in]

    return np.asarray(hca_out, dtype=object), domain


if __name__ == "__main__":
    # File emplacement.
    path: str = "data/hca.out"

    # Convert the list into a numpy array and print it.
    print(parse_hca_file(path, False))
