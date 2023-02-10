"""Parse a hydrophobic cluster (HC) database, which have to be in a `.csv`
format.
"""


__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "03/02/2023"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# [T]
from tqdm import tqdm


def parse_hcdb(path: str) -> "dict[int: str]":
    """Parse the HC database to get regular secondary structure.
    
    Parameters
    ----------
    path : str
        The HC database file's path.

    Returns
    -------
    dict[int: str]
        A dictionary containing in key a Peitsch code and in value a regular
        secondary structure.
    """
    hcdb_dict: "dict[int: str]" = {}

    # Read the database.
    with open(path, "r", encoding="utf-8") as file:
        for line in tqdm(file, "      PARSING DATABASE"):
            # Skipping the first line.
            if line[0] == "#":
                continue

            # To break the `.csv`.
            read_line: "list[str]" = line.strip().split(",")
            hcdb_dict[int(read_line[0])] = read_line[2]

    return hcdb_dict

if __name__ == "__main__":
    print(parse_hcdb("data/HCDB_summary.csv"))
