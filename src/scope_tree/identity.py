"""
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "045/04/2023"
__version__ = "0.0.2"
__copyright__ = "CC BY-SA"



# ==================================================

# To skip all warnings link to numpy module version.

# [W]
from warnings import simplefilter
simplefilter("ignore", DeprecationWarning)

# ==================================================


# [ FULL IMPORT ]
# [I]
import igviz
# [N]
import networkx as net
import numpy as np


# [ PARTIAL IMPORT ]
# [B]
from Bio import Entrez, pairwise2
from Bio.Seq import Seq
# [M]
from matplotlib import colormaps
# [P]
import plotly.graph_objects as go
# [S]
from statistics import median
from sys import path
# [T]
from tqdm import tqdm


# [ EXTERNAL IMPORT ]
# Checking parent directory.
path.append("src/embeddings/")

# [C]
from context_analyzer import PairewiseContextAnalyzer, center_context


def read_fasta(
    path: str,
    domain_list: "list[str]"
) -> dict[str, str]:
    """Parse a given `.fasta` file and return a dictionary with domains and
    their associated sequence.

    Parameters
    ----------
    path : `str`
        The `.fasta` file path.
    domain_list : `list[str]`
        The domain list of a given Peitsch code.

    Returns
    -------
    `dict[str, str]`
        The dictionary with domains as `keys()` and their sequence as
        `values()`.
    """
    domain: "dict[str, str]" = {}
    domain_name: str = ""
    
    with open(path, "r", encoding="utf-8") as file:
        for line in tqdm(list(file), "   PARSING `.FASTA` FILE"):
            if line[0] == ">":
                if domain_name != "":
                    domain[domain_name] = sequence.upper()

                domain_name = "d" + line.split()[0][2:]
                sequence: str = ""
                continue

            if domain_name not in domain_list:
                domain_name = ""
                continue

            sequence += line.strip()

        if domain_name in domain_list:
            domain[domain_name] = sequence.upper()

    return domain


def extract_domain(path: str, code: int) -> "list[str]":
    """Get a domain list from a given Peitsch code.

    Parameters
    ----------
    path : `str`
        Path to pyHCA output file.
    code : `int`
        The Peitsch code.

    Returns
    -------
    `list[str]`
        The list of extracted domains.
    """
    last_power = 0
    cluster = ""

    # To translate a Peitsch code into a hydrophobic cluster.
    while code > 0:
        power = 0

        # When 2 ** power is greater than the code, power - 1 is the index
        # where we can put a 1 into the hydrophobic cluster list/text.
        while 2 ** power < code:
            power += 1

        # When we reach the last position, we don't have to do `-1` to the
        # power.
        if power != 0:
            power -= 1

        # Done only when the first `1` is add to the cluster.
        if cluster != "":
            cluster += "0" * (last_power - power - 1)

        last_power = power
        code -= 2 ** power
        cluster += "1"

    domain_list: "list[str]" = []
    context_dict: "dict[str : list]" = {}
    in_domain: bool = False
    domain: str = ""

    # Parse the file to check if a given cluster is in a domain.
    with open(path, "r", encoding="utf-8") as file:
        for line in tqdm(list(file), "    PARSING pyHCA FILE"):
            # Only checking domain ID and cluster line.
            if line[0] != ">" and not line.startswith("cluster"):
                continue

            # Domaine line.
            if line[0] == ">":
                if in_domain:
                    # Add the centered context to the dictionary.
                    context_dict[domain] = list(center_context(
                        context=np.array([context]),
                        window=10,
                        center=cluster,
                        add_gap=False,
                        keep_center=False
                    ))

                domain = "d" + line.split()[0][2:]
                context: "list[str]" = []
                in_domain = False
                continue

            hc: str = line.strip().split()[-1]

            # Skip to small clusters.
            if len(hc) <= 2:
                continue
            context += [hc]

            # Cluster line: if we found the exact same cluster, add it to our
            # list.
            if cluster != hc:
                continue

            in_domain |= True
            domain_list += [domain]

    domain_list = list(set(domain_list))
    domain_list.sort()
    
    return domain_list


def leaf_identity(
    domain: "dict[str, str]",
):
    len_domain: int = len(domain)
    value_list: "list[int]" = list(range(len_domain))
    key_list: "list[int]" = list(domain)
    
    matrix_index: "dict[str, int]" = dict(zip(domain, value_list))

    identity_matrix: np.ndarray = np.zeros((len_domain, len_domain),
                                     dtype=np.dtype("i2"))

    for i, d_i in enumerate(tqdm(key_list[:-1], "        COMPUTING MATRIX")):
        for d_j in key_list[i + 1:]:
            align_seq = pairwise2.format_alignment(
                *pairwise2.align.localxs(Seq(domain[d_i]), Seq(domain[d_j]),
                                         -6, -1)[0],
                full_sequences=True
            ).split("\n")[1]

            identity: int = round(align_seq.count("|") / len(align_seq) * 100)

            identity_matrix[matrix_index[d_i]][matrix_index[d_j]] = identity
            identity_matrix[matrix_index[d_j]][matrix_index[d_i]] = identity

    return identity_matrix


if __name__ == "__main__":
    domain_list: "list[str]" = extract_domain(
        path="data/pyHCA_SCOPe_30identity_globular.out",
        code=9
    )

    domain: "dict[str, str]" = read_fasta(
        path="data/SCOPe_2.08_95identity_globular.fasta",
        domain_list=domain_list
    )

    identity_matrix = leaf_identity(domain)

    print(np.sum(identity_matrix > 30))
