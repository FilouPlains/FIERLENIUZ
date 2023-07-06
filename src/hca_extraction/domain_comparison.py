"""Generates embedding vector for a given corpus.
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "29/06/2023"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


#  [N]
import numpy as np
# [T]
from tqdm import tqdm

# [O]
from os.path import exists
# [S]
from sys import path as s_path, exit as s_exit
# [T]
from textwrap import wrap

# [ EXTERNAL IMPORT ]
# Checking parent directory.
s_path.append("src/embeddings/")

# [C]
from context_analyzer import PairewiseContextAnalyzer, center_context


def __translate(code: "int") -> str:
    """Translate a hydrophobic cluster into a Peitsch code.

    Parameters
    ----------
    code : int
        The Peitsch code.

    Returns
    -------
    str
        The hydrophobic cluster.
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

    return cluster


def __format_end_line(line: str, length: int = 80) -> str:
    """To flush text with a specific formatting.

    Parameters
    ----------
    line : `str`
        The line to flush and to format.
    length : `int`, optional
        The number of characters before flushing the line, by default 80.

    Returns
    -------
    `str`
        The format line, as a string.
    """
    # Flush the line as a list.
    line: list = wrap(line, width=length)
    text: str = ""

    # Addend things at the end of the line.
    for l_i in line:
        text += f"┃ {l_i:<76} ┃\n"

    # Return everything but the last character, which is a '\n'.
    return text[:-1]


def parse_hc(
    *code,
    path: str,
    scope: str,
    max_level: int = 4,
    window: int = 10,
    treshold: float = 0.5,
    skip_same: bool = False,
    print_results: bool = True
) -> list:
    """Parse differents HC, for a given database, and output sequences that
    share low ordered context diversity.

    Parameters
    ----------
    *code: `int`
        The Peitsch code to compared. Give as mush as you want, but do not
        forget to give after the parameters names, like so:

        >>> parse_hc(
        >>>     # Give here all Peitsch code that you want.
        >>>     105, 165, 109, ...
        >>>     # Do not forget here do give parameters names.
        >>>     path="database.out",
        >>>     treshold=0.5,
        >>>     skip_same=True,
        >>>     print_results=True
        >>> )

    path : `str`
        The database path. It needs to be in a pyHCA`.out` format (unchecked
        here).
    path : `str`
        The path to the SCOPe classification file.
    max_level : `int`, optional
        Integer between 1 and 4. It is used to define the depth in the SCOPe
        classification tree that have to be taken in consideration. 1 the
        class, 2 the fold, 3 the superfammily and 4 the familly, by default 4.
    window : `int`, optional
        Window size to evalute the context diversity, by default 10.
    treshold : `float`, optional
        When value with a lower or equal that the given treashold, they are
        output, by default 0.5.
    skip_same : `bool`, optional
        Skip when two differents HC are on the same database., by default False.
    print_results : `bool`, optional
        Printing at each step the result? If `True`, print the result, by 
        default `True`.

    Returns
    -------
    `list`
        The obtained result.
    """
    if max_level > 4 or max_level < 1:
        s_exit("\n[Err##] SCOPe depth have to be equal to 1, 2, 3 or 4.")

    classification: "dict[str: str]" = {}

    # Parse a given SCOPe classification data file.
    with open(scope, "r", encoding="utf-8") as file:
        for line in tqdm(list(file), "    PARSING SCOPe FILE"):
            # Skip the comment lines.
            if line[0] == "#":
                continue

            split_line: "list[str]" = line.split()

            # Skip the line containing other things than domains.
            if split_line[3] == "-":
                continue

            if split_line[2][0] > "d":
                break

            s_class: str = ".".join(split_line[2].split(".")[:max_level])

            # Adding a domain to the classification dictionary.
            classification[split_line[3]] = s_class

    len_code: int = len(code)

    # Check that
    code: list = filter(lambda x: (isinstance(x, int) and x % 2 == 1), code)
    code: np.ndarray = np.array(list(code))

    if len(code) != len_code:
        s_exit("\n[Err##] Using odd Peitsch code or Peitsch code that are not "
               "integer.")

    if not exists(path):
        s_exit("\n[Err##] Given path does not exist.")

    code_corpus: "dict[int: dict]" = {}

    with open(path, "r", encoding="utf-8") as file:
        file: list = list(file)
        for code_i in tqdm(code, "PARSING HC CORPUS"):
            sentence: "list[str]" = []
            corpus: "dict[str: list]" = {}
            hc: str = __translate(code_i)

            for line in file:
                line: list = line.strip().split()

                if len(line) < 1:
                    continue

                if line[0][0] == ">":
                    if hc in sentence:
                        corpus[d_name] = list(center_context(
                            context=np.array([sentence]),
                            window=window,
                            center=hc,
                            add_gap=False,
                            keep_center=True
                        )[0])

                    sentence = []
                    d_name: str = "d" + line[0][2:]
                elif line[0].startswith("cluster"):
                    sentence += [line[-1]]
                else:
                    continue

            code_corpus[code_i] = corpus

    data: list = [(("domaine_name_i", "code_peitsch_i", "hc_sequence_i"),
                   ("domaine_name_j", "code_peitsch_j", "hc_sequence_j"))]

    for i, code_i in enumerate(code[:-1]):
        for code_j in code[i + 1:]:
            for d_m, s_m in tqdm(code_corpus[code_i].items(), " PARSING HC CODES"):
                for d_n, s_n in code_corpus[code_j].items():
                    same_d: bool = d_m == d_n and skip_same
                    same_c: bool = classification[d_m] == classification[d_n] \
                        and skip_same

                    if same_d or same_c:
                        continue

                    Context: object = PairewiseContextAnalyzer(s_m, s_n)
                    Context.onp_sequence_comparison()
                    diversity: float = Context.distance[0]

                    if diversity <= treshold:
                        data += [((d_m, code_i, s_m), (d_n, code_j, s_n))]

                        if not print_results:
                            continue

                        print("┏" + "━" * 78 + "┓")

                        line = (f"\033[1m\033[91mDiversity = {diversity:.3f}"
                                "\033[0m")
                        print(f"┃ {line:<89} ┃")

                        print("┃" + " " * 78 + "┃")

                        line = f">{d_m} - {code_i}"
                        print(f"┃ {line:<76} ┃")
                        print(__format_end_line(' '.join(s_m), 76))

                        line = f">{d_n} - {code_j}"
                        print(f"┃ {line:<76} ┃")
                        print(__format_end_line(' '.join(s_n), 76))

                        print("┗" + "━" * 78 + "┛\n")

    return data


if __name__ == "__main__":
    print(parse_hc(
        105,
        165,
        path="data/REDUNDANCY_DATASET/cd-hit_90.out",
        scope="data/SCOPe_2.08_classification.txt",
        max_level=2,
        treshold=0.4,
        skip_same=True,
        print_results=True
    ))
