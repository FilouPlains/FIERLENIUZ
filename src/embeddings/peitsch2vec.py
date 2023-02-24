"""Generates embedding vector for a given corpus.

Usage
-----
    Enter in the terminal `python3 src/embeddings/peitsch2vec.py --help` to get
    the detailed help generate by `argparse`.
"""


__authors__ = ["DUPRAT Élodie", "ROUAUD Lucas"]
__contact__ = ["elodie.duprat@sorbonne-universite.fr",
               "lucas.rouaud@gmail.com"]
__date__ = "03/02/2023"
__version__ = "1.5.0"
__copyright__ = "CC BY-SA"

# ==================================================

# To skip all warnings link to numpy module version.

# [W]
from warnings import simplefilter
simplefilter("ignore", UserWarning)

# ==================================================

# [G]
import gensim
# [N]
import numpy as np
# [O]
import os

# [D]
from datetime import datetime
# [H]
from hcdb_parser import parse_hcdb
from hca_reader import parse_hca_file
# [P]
from arg_parser import parsing
from peitsch import Peitsch
# [S]
from sys import path
# [T]
from tqdm import tqdm

# Checking parent directory.
path.append("src/scope_tree/")

from scope_tree import Scope


def matrix_normed(matrix: object) -> object:
    """Return a product normalized matrix.

    Parameters
    ----------
    matrix : object
        The matrix to be normalized.

    Returns
    -------
    object
        A product of the normalized matrix.
    """
    # Create a normalized vector.
    norm_vect: object = np.linalg.norm(matrix, axis=1)

    # Repeat the vector to have a matrix.
    norm_matrix: object = np.array(list(norm_vect) * norm_vect.shape[0])
    norm_matrix = norm_matrix.reshape((matrix.shape[0], matrix.shape[0]))
    
    # Create a product matrix (between normal and transpose one).
    return np.multiply(norm_matrix, np.rot90(norm_matrix, k=3))


if __name__ == "__main__":
    # =====================
    #
    # PROGRAM INITILIZATION
    #
    # =====================

    introduction: str = """
    ███████╗██╗███████╗██████╗ ██╗     ███████╗███╗   ██╗██╗██╗   ██╗███████╗
    ██╔════╝██║██╔════╝██╔══██╗██║     ██╔════╝████╗  ██║██║██║   ██║██╔════╝
    █████╗  ██║█████╗  ██████╔╝██║     █████╗  ██╔██╗ ██║██║██║   ██║███████╗
    ██╔══╝  ██║██╔══╝  ██╔══██╗██║     ██╔══╝  ██║╚██╗██║██║██║   ██║╚════██║
    ██║     ██║███████╗██║  ██║███████╗███████╗██║ ╚████║██║╚██████╔╝███████║
    ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝╚═╝ ╚═════╝ ╚══════╝
    """

    print(introduction)
    arg: "dict[str: str|int]" = parsing()

    # Get data.
    hca_out, domain_data = parse_hca_file(arg["input"], arg["segment"])
    peitsch_manip: object = Peitsch(parse_hcdb(arg["hcdb"]))
    scope_manip: object = Scope(arg["scope"])

    # Getting all PEITSCH data.
    for i, line in enumerate(tqdm(hca_out, "  GETTING PEITSCH DATA")):
        peitsch_manip += line[2]

        # Getting Peitsch code.
        code: int = peitsch_manip.peitsch[i]
        
        # To read each line to get when we change of domain/segment.
        if arg["segment"]:
            read_domain: str = f"{line[0]}_{line[1]}"
        else:
            read_domain: str = line[0]

        peitsch_manip.add_domain(domain_data[read_domain] + [line[0]], code)

    corpus: "list[list[str]]" = []
    sentence: "list[str]" = []
    already_parse_code: "list[str]" = []
    shift: str = ""

    # =================
    #
    # MAKING THE CORPUS
    #
    # =================

    for i, line in enumerate(tqdm(hca_out, "         MAKING CORPUS")):
        # Getting Peitsch code.
        code: int = peitsch_manip.peitsch[i]
        # Getting characteristics list.
        charact: "list" = peitsch_manip.characteristic[code]

        # Checking the hydrophobic cluster length.
        if charact[0].shape[0] <= arg["minlen"]:
            continue
        # Checking the number of total cluster.
        if arg["mintf"] is not None:
            if charact[4] <= arg["mintf"]:
                continue
        if arg["maxtf"] is not None:
            if charact[4] >= arg["maxtf"]:
                continue

        # Initialisation.
        if shift == "":
            # To change sentence by segment or by domain.
            if arg["segment"]:
                shift: str = f"{line[0]}_{line[1]}"
            else:
                shift: str = line[0]

        # To read each line to get when we change of domain/segment.
        if arg["segment"]:
            read_domain: str = f"{line[0]}_{line[1]}"
        else:
            read_domain: str = line[0]

        # Adding to the "sentence" when we have the same domain/segment.
        if shift == read_domain:
            sentence += [f"{code}"]
        # Adding to the "corpus" when we have a different domain/segment.
        else:
            corpus += [sentence]
            sentence = [f"{code}"]

            # To change sentence by segment or by domain.
            if arg["segment"]:
                shift = f"{line[0]}_{line[1]}"
            else:
                shift = line[0]
        
        if code not in already_parse_code:
            already_parse_code += [code]
            peitsch_manip.add_global_score(code, scope_manip)

    # Final sentence addition.
    corpus += [sentence]

    # =======================
    #
    # TRAINING WORD2VEC MODEL
    #
    # =======================

    # Build model.
    peitsch2vec = gensim.models.Word2Vec(
        corpus,
        sg=1,
        seed=1,
        workers=arg["cpu"],
        vector_size=arg["size"],
        min_count=arg["mintf"],
        window=arg["window"],
        sample=arg["sample"]
    )

    # Train the the network.
    peitsch2vec.build_vocab(corpus)
    peitsch2vec.train(
        corpus,
        total_examples=hca_out.shape[0],
        epochs=arg["epochs"]
    )

    # ===========
    #
    # SAVING DATA
    #
    # ===========

    # Get actual data and time.
    date: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ======================
    # SAVE THE COMPUTE MODEL
    # ======================
    model_path: str = os.path.join(arg["output"], f"model_{date}.w2v")
    # peitsch2vec.save(model_path)

    # ===================
    # SAVE THE WORDS DATA
    # ===================
    word_data_path: str = os.path.join(arg["output"], f"embedding_{date}.npy")
    # Convert the embeddings into float64.
    word_data: object = np.array(peitsch2vec.wv.vectors.astype("float64"),
                                 dtype="float64")
    # np.save(word_data_path, word_data, allow_pickle=True)

    # ========================
    # SAVE THE CHARACTERISTICS
    # ========================
    charact_list: "list[list]" = []

    for key in tqdm(peitsch2vec.wv.index_to_key, "SAVING CHARACTERISTICS"):
        charact_list += [[key] + peitsch_manip.characteristic[int(key)]]

    charact_data_path: str = os.path.join(arg["output"],
                                          f"characteristics_{date}.npy")
    
    # Creating the numpy arrays.
    charact_data: object = np.array(charact_list, dtype=object)
    # np.save(charact_data_path, charact_data, allow_pickle=True)

    # ======================
    # SAVE THE COSINE MATRIX
    # ======================
    # Create a matrix with embedding only.
    matrix_embedding: object = np.dot(peitsch2vec.wv.vectors,
                                      peitsch2vec.wv.vectors.T)
    
    # Create a matrix with cosine distance vectors and save it.
    cosine_path: str = os.path.join(arg["output"], f"matrix_cosine_{date}.npy")
    matrix_cosine: object = np.divide(matrix_embedding,
                                      matrix_normed(peitsch2vec.wv.vectors))
    # np.save(cosine_path, matrix_cosine, allow_pickle=True)
