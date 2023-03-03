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
    DESCRIPTION: str = ("Program to compute 'words embedding' with given "
                        "Peitsch code. An command example, when you are at "
                        "the project's root (FIERLENIUS/) is:\n\n    "
                        "$ python3 peitsch2vec.py -i data/hace.out -o "
                        "data/mintf_30/ -d data/HCDB_summary.csv\n\nlegend:\n  "
                        "int: Integer.\n  [type|value]: Type of the input "
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
        metavar="[FILE]['.out']",
        help="[[MANDATORY]]\n    > A pyHCA segmentation results file."
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        required=True,
        type=str,
        metavar="[DIRECTORY]",
        help="[[MANDATORY]]\n    > A folder where the results will be stored."
    )

    parser.add_argument(
        "-d",
        "--hcdb",
        dest="hcdb",
        required=True,
        type=str,
        metavar="[FILE]['.csv']",
        help="[[MANDATORY]]\n    > The HC database path."
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

    # == OPTIONAL.
    parser.add_argument(
        "--epochs",
        default=5,
        type=int,
        metavar="[int|5]",
        help="    > Number of epochs."
    )

    parser.add_argument(
        "--mintf",
        default=30,
        type=int,
        metavar="[int|30]",
        help=("    > Maximum term frequency. If 'None', no threshold will be "
              "applied.")
    )

    parser.add_argument(
        "--maxtf",
        default=None,
        type=int,
        metavar="[int|None]",
        help=("    > Maximum term frequency. If 'None', no threshold will be "
              "applied.")
    )

    parser.add_argument(
        "--minlen",
        default=2,
        type=int,
        metavar="[int|2]",
        help="    > Minimum length of a hydrophobic cluster."
    )

    parser.add_argument(
        "--sample",
        default=1e-3,
        type=int,
        metavar="[float|1e-3]",
        help="    > Down sampling setting for frequent words."
    )

    parser.add_argument(
        "--size",
        default=300,
        type=int,
        metavar="[int|300]",
        help="    > Size of the words embeddings vector."
    )

    parser.add_argument(
        "--window",
        default=5,
        type=int,
        metavar="[int|5]",
        help="    > Window size."
    )

    # Check the computer's number of CPU.
    nb_cpu: int = cpu_count()

    parser.add_argument(
        "--cpu",
        default=nb_cpu,
        required=False,
        type=int,
        metavar=f"[int|{nb_cpu}]",
        help=("    > Number of processes, by default set to your number "
              f"of CPU. Here, {nb_cpu} CPU are detected.")
    )

    parser.add_argument(
        "--segment",
        required=False,
        action="store_true",
        help=("    > If used, output sentence by foldable segment and not by "
              "domain. By default this option is inactivated, so 'False' is "
              "store.")
    )

    # Transform the input into a dictionary with arguments as key.
    argument = vars(parser.parse_args())

    # ===============================
    #
    # TESTS IF PARAMETERS ARE CORRECT
    #
    # ===============================
    # Check the input file extension.
    if not argument["input"].endswith(".out"):
        sysexit(f"\n[Err##] The input file '{argument['input']}' extension "
                "is invalid. Please, give a '.out' file.")

    # Check if the input file exists.
    if not os.path.exists(argument["input"]):
        sysexit(f"\n[Err##] The input file '{argument['input']}' does not "
                "exist. Please check this given file.")

    # Check the HCDB file extension.
    if not argument["hcdb"].endswith(".csv"):
        sysexit(f"\n[Err##] The hcdb file '{argument['hcdb']}' extension "
                "is invalid. Please, give a '.csv' file.")

    # Check if the HCDB file exists.
    if not os.path.exists(argument["hcdb"]):
        sysexit(f"\n[Err##] The input file '{argument['hcdb']}' does not "
                "exist. Please check this given file.")

    # Check the SCOPe classification file extension.
    if not argument["scope"].endswith(".txt"):
        sysexit(f"\n[Err##] The hcdb file '{argument['scope']}' extension "
                "is invalid. Please, give a '.txt' file.")

    # Check if the SCOPe classification file exists.
    if not os.path.exists(argument["scope"]):
        sysexit(f"\n[Err##] The input file '{argument['scope']}' does not "
                "exist. Please check this given file.")

    # Check if the output directory exists.
    if not os.path.exists(argument["output"]):
        sysexit(f"\n[Err##] The output directory '{argument['output']}' "
                "does not exist. Please check this given directory.")

    # Check if the input number of CPU is correct.
    if argument["cpu"] > nb_cpu:
        sysexit(f"\n[Err##] Ask number of CPU, {argument['cpu']}, is "
                "superior to the number of CPU of this computer, which "
                f"is {nb_cpu}. Please change the input number.")

    # Check if the input number of CPU is correct.
    if argument["cpu"] <= 0:
        sysexit(f"\n[Err##] Ask number of CPU, {argument['cpu']}, is "
                "inferior or equal to 0. Please change the input number.")

    to_check: "list[str]" = ["epochs", "minlen", "size", "window"]

    if argument["mintf"] is not None:
        to_check += ["mintf"]
    if argument["maxtf"] is not None:
        to_check += ["maxtf"]

    # Check that a bunch of values are > 0.
    for key in to_check:
        if argument[key] <= 0:
            sysexit(f"\n[Err##] The given arguments '{key}' is incorrect. "
                    "You have to give an int greater strictly than 0.")

    if argument["mintf"] is not None and argument["maxtf"] is not None:
        if argument["mintf"] >= argument["maxtf"]:
            sysexit("\n[Err##] The given arguments for '--mintf' "
                    f"(= {argument['mintf']}) and for '--maxtf' (= "
                    f"{argument['maxtf']}) are incorrect. You have to give an "
                    "int so the maximum is strictly greater than the "
                    "minimum.")

    return argument


if __name__ == "__main__":
    parser: "dict[str: ]" = parsing()

    for key in parser:
        print(f"{parser[key]=}")
