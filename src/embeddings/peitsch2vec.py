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
__version__ = "1.2.0"
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
# [T]
from tqdm import tqdm


if __name__ == "__main__":
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
    hca_out: object = parse_hca_file(arg["input"])
    peitsch_manip: object = Peitsch(parse_hcdb("data/HCDB_summary.csv"))

    # Getting all PEITSCH data.
    for line in tqdm(hca_out, "  GETTING PEITSCH DATA"):
        peitsch_manip += line[2]

    corpus: "list[list[str]]" = []
    sentence: "list[str]" = []
    shift: str = ""

    for i, line in enumerate(tqdm(hca_out, "         MAKING CORPUS")):
        # Getting Peitsch code.
        code: int = peitsch_manip.peitsch[i]
        # Getting characteristics list.
        charact: "list" = peitsch_manip.characteristic[code]

        # Checking the hydrophobic cluster length.
        if charact[0].shape[0] <= arg["minlen"]:
            continue
        # Checking the number of total cluster.
        if charact[-1] <= arg["mintf"] or charact[-1] >= arg["maxtf"]:
            continue

        # Initialisation.
        if shift == "":
            # To change sentence by segment or by domain.
            if arg["segment"]:
                shift = f"{line[0]}_{line[1]}"
            else:
                shift = line[0]
        # Adding to the "sentence" when we have the same domain/segment.
        if shift == line[0]:
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

    # Final sentence addition.
    corpus += [sentence]

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

    # Get actual data and time.
    date: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save the compute model.
    model_path: str = os.path.join(arg["output"], f"model_{date}.w2v")
    peitsch2vec.save(model_path)

    # Save the words data.
    word_data_path: str = os.path.join(arg["output"], f"embedding_{date}.npy")
    # Convert the embeddings into float64.
    word_data: object = np.array(peitsch2vec.wv.vectors.astype("float64"),
                                 dtype="float64")
    np.save(word_data_path, word_data, allow_pickle=True)

    charact_list: "list[list]" = []

    for key in tqdm(peitsch2vec.wv.index_to_key, "SAVING CHARACTERISTICS"):
        charact_list += [[[key] + peitsch_manip.characteristic[int(key)]]]

    # Save the words data.
    charact_data_path: str = os.path.join(arg["output"],
                                          f"characteristics_{date}.npy")
    # Convert the embeddings into float64.
    charact_data: object = np.array(charact_list)
    np.save(charact_data_path, charact_data, allow_pickle=True)
