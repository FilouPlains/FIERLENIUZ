"""Generates embedding vector for a given corpus.
"""


__authors__ = ["DUPRAT Élodie", "ROUAUD Lucas"]
__contact__ = ["elodie.duprat@sorbonne-universite.fr",
               "lucas.rouaud@gmail.com"]
__date__ = "03/02/2023"
__version__ = "1.1.0"
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
from embeddings.arg_parser import parsing
from peitsch import Peitsch
# [T]
from tqdm import tqdm


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
        min_count=arg["mintf"],
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
