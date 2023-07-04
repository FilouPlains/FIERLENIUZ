"""Parse given arguments and do some test to see if there are correct or not.
"""


__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "03/02/2023"
__version__ = "1.0.1"
__copyright__ = "CC BY-SA"


# [A]
import argparse
# [O]
import os

# [M]
from multiprocessing import cpu_count
# [S]
from sys import exit as sysexit


def parsing():
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
    # Description of the program given when the help is cast.
    DESCRIPTION: str = ("Program to convert a '.fasta' file into a '.out' "
                        "file, which is a pyHCA segmentation results file. An "
                        "example of command, when you are at the project's "
                        "root (FIERLENIUS/) is:\n\n    $ python3 src/hca_"
                        "extraction/hca_extraction.py -i data/REDUNDANCY_"
                        "DATASET/cd-hit_30.fasta -o data/REDUNDANCY_DATASET/"
                        "cd-hit_30.out -s data/SCOPe_2.08_classification.txt"
                        "\n\nLegend:\n    [type|value]: Type of the input "
                        "required, follow by the default value. So if this "
                        "optional arguments is not used, 'value' will be "
                        "chosen.")

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
        metavar="[FILE]['.fasta']",
        help=("[[MANDATORY]]\n    > A '.fasta' file containing all sequences "
              "to transform into hydrophobic cluster.")
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        required=True,
        type=str,
        metavar="[DIRECTORY]['.out']",
        help="[[MANDATORY]]\n    > A pyHCA-like segmentation results file."
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

    # ===============================
    #
    # TESTS IF PARAMETERS ARE CORRECT
    #
    # ===============================
    # Check the input file extension.
    if not argument["input"].endswith(".fasta"):
        sysexit(f"\n[Err##] The input file '{argument['input']}' extension "
                "is invalid. Please, give a '.fasta' file.")

    # Check if the input file exists.
    if not os.path.exists(argument["input"]):
        sysexit(f"\n[Err##] The input file '{argument['input']}' does not "
                "exist. Please check this given file.")

    # Check the SCOPe classification file extension.
    if not argument["scope"].endswith(".txt"):
        sysexit(f"\n[Err##] The hcdb file '{argument['scope']}' extension "
                "is invalid. Please, give a '.txt' file.")

    # Check if the SCOPe classification file exists.
    if not os.path.exists(argument["scope"]):
        sysexit(f"\n[Err##] The input file '{argument['scope']}' does not "
                "exist. Please check this given file.")

    # Check the output file extension.
    if not argument["output"].endswith(".out"):
        sysexit(f"\n[Err##] The input file '{argument['output']}' extension "
                "is invalid. Please, give a '.out' file.")

    # Check if the output directory exists.
    if os.path.exists(argument["output"]):
        sysexit(f"\n[Err##] The output directory '{argument['output']}' "
                "does exist. Please give an another directory/file name.")

    return argument


if __name__ == "__main__":
    parser: "dict[str: ]" = parsing()

    for key in parser:
        print(f"{parser[key]=}")
