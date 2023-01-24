"""Program used to transform the whole `.hca` file into a numpy array
file (`.npy`).
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "24/01/2023"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# [N]
import numpy as np

# [T]
from tqdm import tqdm

if __name__ == "__main__":
    # File emplacement.
    path: str = "embeddings/hca.out"
    hca_out: object = []

    # Read the file.
    with open(path, "r", encoding="utf-8") as file:
        domain_name: str = ""

        for line in tqdm(file, desc="PARSING FILE"):
            # To keep the domain's ID.
            if line.startswith(">"):
                domain_name = line.split()[0][1:]

                # Else, the domain name are not seperated from their first
                # position.
                if domain_name[-1] != "_":
                    domain_name += "_"

                continue
            # Skip all first comments or domain lines.
            elif not domain_name or line.startswith("domain"):
                continue

            # Transform the line into a list.
            split_line: "list[str]" = line.split()
            # Keep the domain's ID and a numpy array as a bool type.
            read_in: "list[str]" = [domain_name + split_line[1],
                                    np.array([*split_line[-1]], dtype=bool)]

            hca_out += [read_in]

    # Convert the list into a numpy array and save it.
    hca_out = np.asarray(hca_out, dtype=object)
    np.save("embeddings/hca.npy", hca_out, allow_pickle=True)
