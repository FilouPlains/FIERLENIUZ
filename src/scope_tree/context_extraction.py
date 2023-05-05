"""This module contains an object to manipulate SCOPe classification.
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "04/04/2023"
__version__ = "2.0.0"
__copyright__ = "CC BY-SA"


# [ FULL IMPORT ]
import numpy as np
import sys

# [ PARTIAL IMPORT ]
# [D]
from datetime import datetime
# [O]
from os import popen
from sys import path
# [T]
from tqdm import tqdm


# [ EXTERNAL IMPORT ]
# Checking parent directory.
path.append("src/embeddings/")

# [C]
from context_analyzer import PairewiseContextAnalyzer, center_context


class ScopeTree:
    """An object to manipulate SCOPe classification.
    """

    def __init__(
        self,
        path: str,
        index_dict: "dict[str : int]"
    ) -> None:
        """Instantiate a `Scope` object.

        Parameters
        ----------
        path : `str`
            A path to a SCOPe classification data file.
        index_dict : `dict[str : int]`
            Index liking domain (as keys) to index in the matrix (as int).
        """
        # To go from domain to SCOPe classification code.
        self.classification: "dict[str: str]" = {}

        # To link SCOPe code classification (as key) to their index in other
        # list.
        self.index: "dict[str : int]" = {"0": 0}

        # Set up attributes with given parameters.
        self.index_dict: "dict[str : int]" = index_dict

        # To get the index list ordered in function of which domain we compare.
        self.matrix_index: "list[list[int]]" = [[]]

        # Parse a given SCOPe classification data file.
        with open(path, "r", encoding="utf-8") as file:
            for line in tqdm(list(file), " PARSING SCOPe FILE"):
                # Skip the comment lines.
                if line[0] == "#":
                    continue

                split_line: "list[str]" = line.split()

                # Skip the line containing other things than domains.
                if split_line[3] == "-":
                    continue

                # Adding a domain to the classification dictionary.
                self.classification[split_line[3]] = split_line[2]

    def __iadd__(self, domain: str) -> object:
        """Add a domain to the network graph.

        Parameters
        ----------
        domain : `str`
            The domain to add.

        Returns
        -------
        object
            self.
        """
        # To check if the domain is in the classification dictionary.
        if domain not in self.classification or domain not in self.index_dict:
            print(f"[Warn##] Domain [{domain}] is not in the SCOPe "
                  "classification dictionary, skipping...")
            return self

        # Adding parameters to the original nodes.
        self.matrix_index[0] += list(self.index_dict[domain])

        # Check a level.
        node: str = self.classification[domain][0]

        # Is this a new node to add ?
        if node not in self.index:
            # Setting index to extract context values from matrix.
            self.matrix_index += [list(self.index_dict[domain])]
        else:
            # Adding index to extract context values from matrix.
            self.matrix_index[self.index[node]] += list(
                self.index_dict[domain]
            )

        return self


def get_domain(path: str, code: int) -> "tuple":
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
        for line in file:
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

    length: int = 0

    # To get the full length of the number of context per domain, because we
    # have a list of list.
    for content in context_dict.values():
        length += len(content)

    keys: "list[str]" = list(context_dict.keys())

    # Set up the futur context matrix.
    order_matrix: np.ndarray = np.zeros((length, length), dtype=float)
    unorder_matrix: np.ndarray = np.zeros((length, length), dtype=float)

    index_dict: "dict[str, int]" = {}
    domain_dict: "dict[str, int]" = {}

    index: int = 0

    # Loop in the half matrix.
    for shift_i, i in enumerate(tqdm(keys[:-1], " MATRIX COMPUTATION")):
        for j in keys[shift_i + 1:]:
            order: "list[float]" = []
            unorder: "list[float]" = []

            # The whole context (domain 1 vs domain 2).
            context: "list[str]" = context_dict[i] + context_dict[j]
            # Domains names linked to `context` (number and order).
            d_cont: "list[str]" = [i] * len(context_dict[i]) \
                + [j] * len(context_dict[j])
            # ID for the domain linked to `context` (number and order).
            n_cont: "list[int]" = list(range(len(context_dict[i]))) \
                + list(range(len(context_dict[j])))

            # Loop in the half matrix of the context in the domain. We do like
            # so as far as we can have multiple context in a domain.
            for pos_a, a in enumerate(context[:-1]):
                if len(a) <= 0:
                    continue

                # Generate a unique key which take in consideration the domain
                # and the multiple context per domain.
                dict_key_a: str = f"{d_cont[pos_a]}_{n_cont[pos_a]}"

                # Add the key to the dictionary.
                if dict_key_a not in index_dict:
                    index_dict[dict_key_a] = index
                    index += 1

                i_a: int = index_dict[dict_key_a]

                # Add matrix index.
                if d_cont[pos_a] not in domain_dict:
                    domain_dict[d_cont[pos_a]] = [i_a]
                else:
                    domain_dict[d_cont[pos_a]] += [i_a]

                for pos_b, b in enumerate(context[pos_a + 1:]):
                    if len(b) <= 0:
                        continue

                    pos_b += 1 + pos_a

                    # Generate a unique key which take in consideration the
                    # domain and the multiple context per domain.
                    dict_key_b: str = f"{d_cont[pos_b]}_{n_cont[pos_b]}"

                    # Add the key to the dictionary.
                    if dict_key_b not in index_dict:
                        index_dict[dict_key_b] = index
                        index += 1

                    i_b: int = index_dict[dict_key_b]

                    # Add matrix index.
                    if d_cont[pos_b] not in domain_dict:
                        domain_dict[d_cont[pos_b]] = [i_b]
                    else:
                        domain_dict[d_cont[pos_b]] += [i_b]

                    # Compute order and unorder context.
                    Context: object = PairewiseContextAnalyzer(a, b)
                    Context.compute_distance()

                    order = Context.distance[0]
                    unorder = Context.distance[1]

                    order_matrix[i_a][i_b] = order
                    order_matrix[i_b][i_a] = order
                    unorder_matrix[i_a][i_b] = unorder
                    unorder_matrix[i_b][i_a] = unorder

    # Return a list of unique elements, as far as a same Peitsch code can be
    # multiple time in the same domain.
    return domain_list, [order_matrix, unorder_matrix, domain_dict]


if __name__ == "__main__":
    redundancy: int = int(sys.argv[1])

    characteristic: object = np.transpose(np.load(
        popen(f"ls data/peitsch2vec/redundancy/{redundancy}_percent_redundancy"
              "/characteristics*.npy").readlines()[0].strip(),
        allow_pickle=True
    ))

    peitsch: object = characteristic[0].astype("int64")

    date: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path: str = ("/home/lrouaud/Téléchargements/redundancy_"
                      f"{redundancy}_context_conservation_{date}.csv")
    save_path="/home/lrouaud/Téléchargements/redundancy.csv"

    with open(save_path, "w", encoding="utf-8") as file:
        file.write("PEITSCH_CODE,ALL_ORDER,A_ORDER,B_ORDER,C_ORDER,"
                   "D_ORDER,ALL_UNORDER,A_UNORDER,B_UNORDER,C_UNORDER,"
                   "D_UNORDER\n")

        # Parse all Peitsch code.
        for i, code in enumerate(tqdm(peitsch, desc=f"PARSING PEISCH_CODE")):
            # Get all data and context values in matrix format.
            domain_list, data_list = get_domain(
                f"data/REDUNDANCY_DATASET/cd-hit_{redundancy}.out",
                code
            )

            # Instantiate a Scope object
            scope_tree: ScopeTree = ScopeTree(
                "data/SCOPe_2.08_classification.txt",
                index_dict=data_list[2]
            )

            # Add domains to the tree.
            for domain in tqdm(domain_list, "PARSING DOMAIN LIST"):
                scope_tree += domain

            # So 1 Peitsch code = 1 line.
            line: list = [[]] * 10

            # Parse all SCOPe class to output distributions in function of them.
            # "0" refers to the tree's root.
            for j, key in enumerate(["0", "a", "b", "c", "d"]):
                if key in scope_tree.index:
                    # Distribution for the order context.
                    m_i: set = list(
                        set(scope_tree.matrix_index[scope_tree.index[key]]))
                    matrix: np.ndarray = data_list[0][m_i, :][:, m_i]
                    x: np.ndarray = matrix[
                        np.triu_indices(matrix.shape[0], k=1)
                    ] * 100

                    if len(list(x)) != 0:
                        line[j] = f"{np.mean(x)}"
                    else:
                        line[j] = f"{1}"
                        line[j + 5] = f"{1}"

                    # Distribution for the unorder context.
                    matrix: np.ndarray = data_list[1][m_i, :][:, m_i]
                    x: np.ndarray = matrix[
                        np.triu_indices(matrix.shape[0], k=1)
                    ] * 100

                    if len(list(x)) != 0:
                        line[j + 5] = f"{np.mean(x)}"

                else:
                    line[j] = f"{1}"
                    line[j + 5] = f"{1}"

            file.write(",".join([f"{code}"] + line))
