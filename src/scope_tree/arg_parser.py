"""Parse given arguments and do some test to see if there are correct or not.
"""


__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "03/02/2023"
__version__ = "1.2.3"
__copyright__ = "CC BY-SA"


# [A]
import argparse
# [O]
import os

# [M]
from multiprocessing import cpu_count
# [R]
from re import search
# [S]
from sys import exit as sysexit


def parsing(is_directory=False):
    """This function call the parser to get all necessary program's arguments.

    Returns
    -------
    dict[str, val**]
        Permit the accessibility to access to all given arguments with their
        values, thanks to a dictionary.
    """
    # ==================
    #
    # CREATE THE PARSER
    #
    # ==================
    if is_directory:
        DESCRIPTION: str =  ("Program used to computed a network plot and "
                             "distribution plot using plotly. An example of "
                             "command, when you are at the project's root "
                             "(FIERLENIUS/) is:\n\n    $ python3 src/scope_"
                             "tree/scope_tree.py -i 105 -s data/SCOPe_2.08_"
                             "classification.txt -o ~/Téléchargements/DOWN"
                             "SAMPLING_1/ -c data/pyHCA_SCOPe_30identity_"
                             "globular.out\n\nOr:\n\n    $ python3 src/scope_"
                             "tree/scope_tree.py -i data/peitsch2vec/"
                             "redundancy/30_percent_redundancy/characteristics"
                             "_2023-04-25_16-16-44.npy -s data/SCOPe_2.08_"
                             "classification.txt -o data/output_plot/ -c data/"
                             "pyHCA_SCOPe_30identity_globular.out\n\nLegend:"
                             "\n  int: Integer.")
    else:
        DESCRIPTION: str = ("Program used to computed the context diversity "
                            "and output it in a '.csv' format.\n\n"
                            "    $ python3 src/scope_tree/context_extraction"
                            ".py -i 105 -s data/SCOPe_2.08_classification.txt "
                            "-o data/REDUNDANCY_DATASET/untreat_redundancy_"
                            "30.csv -c data/pyHCA_SCOPe_30identity_globular"
                            ".out\n\nOr:\n\n    $ python3 src/scope_tree/"
                            "context_extraction.py -i data/peitsch2vec/"
                            "redundancy/30_percent_redundancy/characteristics_"
                            "2023-04-25_16-16-44.npy -s data/SCOPe_2.08_"
                            "classification.txt -o data/REDUNDANCY_DATASET/"
                            "untreat_redundancy_30.csv -c data/pyHCA_SCOPe_"
                            "30identity_globular.out\n\nLegend:\n  int: "
                            "Integer.")


    # Setup the arguments parser object.
    parser: object = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False
    )

    # ==========
    #
    # ARGUMENTS
    #
    # ==========

    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="    > Display this help message, then exit the program."
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"Program version is {__version__}",
        help="    > Display the program's version, then exit the program."
    )

    # == REQUIRED.
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        required=True,
        type=str,
        metavar="[FILE]['.npy'] or [int]",
        help=("[[MANDATORY]]\n    > A Peitch code characteristic file or one "
              "unique Peitch code value.")
    )

    if is_directory:
        parser.add_argument(
            "-o",
            "--output",
            dest="output",
            required=True,
            type=str,
            metavar="[DIRECTORY]",
            help=("[[MANDATORY]]\n    > A folder where the results will be "
                  "stored.")
        )
    else:
        parser.add_argument(
            "-o",
            "--output",
            dest="output",
            required=True,
            type=str,
            metavar="[FILE]['.csv']",
            help=("[[MANDATORY]]\n    > A file were the context diversity "
                  "will be saved, in a '.csv' format. Only SCOPe level 'a', "
                  "'b', 'c', 'd' and 'a to d' are taken in consideration.")
        )

    parser.add_argument(
        "-c",
        "--context",
        dest="context",
        required=True,
        type=str,
        metavar="[FILE]['.out']",
        help=("[[MANDATORY]]\n    > A file containing the context to analyze, "
              "in a pyHCA output format.")
    )

    parser.add_argument(
        "-s",
        "--scope",
        dest="scope",
        required=True,
        type=str,
        metavar="[FILE]['.txt']",
        help="[[MANDATORY]]\n    > A SCOPe classification file."
    )

    # Transform the input into a dictionary with arguments as key.
    argument = vars(parser.parse_args())
    argument["integer"] = False

    # ===============================
    #
    # TESTS IF PARAMETERS ARE CORRECT
    #
    # ===============================
    if search("^[0-9]+$", argument["input"]):
        argument["input"] = int(argument["input"])
        argument["integer"] = True

        if argument["input"] % 2 == 0:
            sysexit(f"[Err##] The given Peitsch code {argument['input']} is "
                    "wrong and should be odd.")
    else:
        # Check the input file extension.
        if not argument["input"].endswith(".npy"):
            sysexit("\n[Err##] Wrong input given. Either:\n    - The input "
                    f"file '{argument['input']}' extension is invalid. Then "
                    "please, give a '.npy' file.\n    - The input Peitsch "
                    "code is bad. Then please, give an odd integer.")

        # Check if the input file exists.
        if not os.path.exists(argument["input"]):
            sysexit(f"\n[Err##] The input file '{argument['input']}' does not "
                    "exist. Please check this given file.")

    # Check the context file extension.
    if not argument["context"].endswith(".out"):
        sysexit(f"\n[Err##] The context file '{argument['context']}' extension "
                "is invalid. Please, give a '.out' file.")

    # Check if the context file exists.
    if not os.path.exists(argument["context"]):
        sysexit(f"\n[Err##] The context file '{argument['context']}' does not "
                "exist. Please check this given file.")

    # Check the SCOPe classification file extension.
    if not argument["scope"].endswith(".txt"):
        sysexit("\n[Err##] The SCOPe classification file "
                f"'{argument['scope']}' extension is invalid. Please, give a "
                "'.txt' file.")

    # Check if the SCOPe classification file exists.
    if not os.path.exists(argument["scope"]):
        sysexit("\n[Err##] The SCOPe classification file "
                "'{argument['scope']}' does not exist. Please check this "
                "given file.")

    if is_directory:
        # Check if the output directory exists.
        if not os.path.exists(argument["output"]):
            sysexit(f"\n[Err##] The output directory '{argument['output']}' "
                    "does not exist. Please check this given directory.")
    else:
        # Check if the output file does not exists.
        if os.path.exists(argument["output"]):
            sysexit(f"\n[Err##] The output directory '{argument['output']}' "
                    "does exist. Please check the output file name.")

    return argument


if __name__ == "__main__":
    parser: "dict[str: ]" = parsing()

    for key in parser:
        print(f"{parser[key]=}")
