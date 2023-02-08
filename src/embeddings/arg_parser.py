"""Parse given arguments and do some test to see if there are correct or not.
"""


__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "03/02/2023"
__version__ = "1.0.0"
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
                        "data/mintf_30/ -d data/HCDB_summary.csv")

    # Setup the arguments parser object.
    parser: object = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # ==========
    #
    # ARGUMENTS
    #
    # ==========
    # == REQUIRED.
    parser.add_argument(
        "-i, --input",
        dest="input",
        required=True,
        type=str,
        help="[[MANDATORY]]\n    ['.out'] A pyHCA segmentation results file."
    )

    parser.add_argument(
        "-o, --output",
        dest="output",
        required=True,
        type=str,
        help="[[MANDATORY]]\n    A folder where the results will be stored."
    )

    parser.add_argument(
        "-d, --hcdb",
        dest="hcdb",
        required=True,
        type=str,
        help="[[MANDATORY]]\n    ['.csv'] The HC database path."
    )

    # == OPTIONAL.
    parser.add_argument(
        "--epochs",
        default=5,
        type=int,
        help="[integer] Number of epochs, by default 5."
    )

    parser.add_argument(
        "--mintf",
        default=30,
        type=int,
        help="[integer] Minimum term frequency, by default 30."
    )

    parser.add_argument(
        "--maxtf",
        default=1000,
        type=int,
        help="[integer] Maximum term frequency, by default 1000."
    )

    parser.add_argument(
        "--minlen",
        default=2,
        type=int,
        help="[integer] Minimum length of a hydrophobic cluster, by default 2."
    )

    parser.add_argument(
        "--sample",
        default=1e-3,
        type=int,
        help=("[float] Down sampling setting for frequent words, by default"
              " 1e-3.")
    )

    parser.add_argument(
        "--size",
        default=300,
        type=int,
        help="[integer] Size of the words embeddings vector, by default 300."
    )

    parser.add_argument(
        "--window",
        default=5,
        type=int,
        help="[integer] Window size, by default 5."
    )

    # Check the computer's number of CPU.
    nb_cpu: int = cpu_count()

    parser.add_argument(
        "--cpu",
        default=nb_cpu,
        required=False,
        type=int,
        help=("[integer] Number of processes, by default set to your number "
              f"of CPU. Here, {nb_cpu} CPU are detected.")
    )

    parser.add_argument(
        "--segment",
        required=False,
        action="store_true",
        help=("If used, output sentence by foldable segment and not by "
              "domain. By default this option inactivated.")
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
        sysexit(f"\n[Err## 1] The input file '{argument['input']}' extension "
                "is invalid. Please, give a '.out' file.")

    # Check if the input file exists.
    if not os.path.exists(argument["input"]):
        sysexit(f"\n[Err## 2] The input file '{argument['input']}' does not "
                "exist. Please check this given file.")

    # Check the input file extension.
    if not argument["hcdb"].endswith(".csv"):
        sysexit(f"\n[Err## 3] The hcdb file '{argument['hcdb']}' extension "
                "is invalid. Please, give a '.csv' file.")

    # Check if the input file exists.
    if not os.path.exists(argument["hcdb"]):
        sysexit(f"\n[Err## 4] The input file '{argument['hcdb']}' does not "
                "exist. Please check this given file.")

    # Check if the output directory exists.
    if not os.path.exists(argument["output"]):
        sysexit(f"\n[Err## 5] The output directory '{argument['output']}' "
                "does not exist. Please check this given directory.")

    # Check if the input number of CPU is correct.
    if argument["cpu"] > nb_cpu:
        sysexit(f"\n[Err## 6] Ask number of CPU, {argument['cpu']}, is "
                "superior to the number of CPU of this computer, which "
                f"is {nb_cpu}. Please change the input number.")

    # Check if the input number of CPU is correct.
    if argument["cpu"] <= 0:
        sysexit(f"\n[Err## 7] Ask number of CPU, {argument['cpu']}, is "
                "inferior or equal to 0. Please change the input number.")

    to_check: "list[str]" = ["epochs", "mintf", "maxtf", "minlen",
                             "size", "window"]

    # Check that a bunch of values are > 0.
    for key in to_check:
        if argument[key] <= 0:
            sysexit(f"\n[Err## 8] The given arguments '{key}' is incorrect. "
                    "You have to give an integer greater strictly than 0.")

    if argument["mintf"] >= argument["maxtf"]:
        sysexit("\n[Err## 9] The given arguments for '--mintf' "
                f"(= {argument['mintf']}) and for '--maxtf' (= "
                f"{argument['maxtf']}) are incorrect. You have to give an "
                "integer so the maximum is strictly greater than the minimum.")

    return argument


if __name__ == "__main__":
    parser: "dict[str: ]" = parsing()

    for key in parser:
        print(f"{parser[key]=}")
