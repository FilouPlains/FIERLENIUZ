"""Generates embedding vector for a given corpus.
"""


__authors__ = ["DUPRAT Élodie", "ROUAUD Lucas"]
__contact__ = ["elodie.duprat@sorbonne-universite.fr",
               "lucas.rouaud@gmail.com"]
__date__ = "30/01/2023"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# ==================================================

# To skip all warnings link to numpy module version.

# [W]
from warnings import simplefilter
simplefilter("ignore", UserWarning)

# ==================================================

# [A]
import argparse
# [G]
import gensim
# [N]
import numpy as np
# [O]
import os
# [P]
import pandas as pd

# [D]
from datetime import datetime
# [G]
from gensim.models import word2vec as w2v
# [H]
from hca_reader import parse_hca_file
# [M]
from multiprocessing import cpu_count
# [P]
from peitsch_translator import peitsch_translator
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
                        "Peitsch code.")

    # Setup the arguments parser object.
    parser: object = argparse.ArgumentParser(description=DESCRIPTION)

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
        help="['.out'] A pyHCA segmentation results file."
    )

    parser.add_argument(
        "-o, --output",
        dest="output",
        required=True,
        type=str,
        help="A folder where the results will be stored."
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
        default=1,
        type=int,
        help="[integer] Minimum term frequency, by default 4."
    )

    parser.add_argument(
        "--model",
        required=False,
        type=str,
        help="['.w2v'] Model file, if exists."
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

    # Check if the output directory exists.
    if not os.path.exists(argument["output"]):
        sysexit(f"\n[Err## 3] The output directory '{argument['output']}' "
                "does not exist. Please check this given directory.")

    if argument["model"]:
        # Check the model file extension.
        if not argument["model"].endswith(".w2v"):
            sysexit(f"\n[Err## 4] The model file '{argument['model']}' "
                    "extension is invalid. Please, give a '.hca' file.")

        # Check if the model file exists.
        if not os.path.exists(argument["model"]):
            sysexit(f"\n[Err## 5] The model file '{argument['model']}' does "
                    "not exist. Please check this given file.")

    # Check if the input number of CPU is correct.
    if argument["cpu"] > nb_cpu:
        sysexit(f"\n[Err## 6] Ask number of CPU, {argument['cpu']}, is "
                "superior to the number of CPU of this computer, which "
                f"is {nb_cpu}. Please change the input number.")

    # Check if the input number of CPU is correct.
    if argument["cpu"] <= 0:
        sysexit(f"\n[Err## 7] Ask number of CPU, {argument['cpu']}, is "
                "inferior or equal to 0. Please change the input number.")

    to_check: "list[str]" = ["epochs", "mintf", "size", "window"]

    # Check that a bunch of values are > 0.
    for key in to_check:
        if argument[key] <= 0:
            sysexit(f"\n[Err## 8] The given arguments '{key}' is correct. You "
                    "have to give a integer greater strictly than 0.")

    return argument


if __name__ == "__main__":
    arg: "dict[str: str|int]" = parsing()

    # Get data.
    hca_out: object = parse_hca_file(arg["input"])
    # Translate hydrophobic clusters into Peitsch code.
    peitsch: "list[list[int]]" = peitsch_translator(hca_out)

    # Build model.
    peitsch2vec = gensim.models.Word2Vec(
        peitsch,
        sg=1,
        seed=1,
        workers=arg["cpu"],
        vector_size=arg["size"],
        min_count=1,
        window=arg["window"],
        sample=arg["sample"]
    )

    # Train the the network.
    peitsch2vec.build_vocab(peitsch)
    peitsch2vec.train(
        peitsch,
        total_examples=hca_out.shape[0],
        epochs=arg["epochs"]
    )

    # Get actual data and time.
    date: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save the compute model.
    model_path: str = os.path.join(arg["output"], f"model_{date}.w2v")
    peitsch2vec.save(model_path)

    # Save the words data.
    word_data_path: str = os.path.join(arg["output"], f"word_data_{date}.npy")
    # Convert the embeddings into float64.
    word_data: object = np.array(peitsch2vec.wv.vectors.astype("float64"),
                                 dtype="float64")
    np.save(word_data_path, word_data, allow_pickle=True)
