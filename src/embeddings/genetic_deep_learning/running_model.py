"""Run a word2vec model.
"""


__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "21/04/2023"
__version__ = "1.0.1"
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

# [C]
from correlation_matrix import ComputeCorrelation
# [G]
from gensim.models.callbacks import CallbackAny2Vec
# [M]
from multiprocessing import cpu_count


def cosine_matrix(embedding: np.ndarray) -> np.ndarray:
    """Return a cosine similarity matrix.

    Parameters
    ----------
    embedding : `np.ndarray`
        The embedding to transform.

    Returns
    -------
    `np.ndarray`
        The cosine similarity matrix.
    """
    # Create a normalized vector.
    norm_vect: np.ndarray = np.linalg.norm(embedding, axis=1)

    # Repeat the vector to have a matrix.
    norm_matrix: np.ndarray = np.array(list(norm_vect) * norm_vect.shape[0])
    norm_matrix = norm_matrix.reshape((embedding.shape[0], embedding.shape[0]))

    # Create the cosine similarity matrix.
    matrix: np.ndarray = np.divide(
        np.dot(embedding, embedding.T),
        np.multiply(norm_matrix, np.rot90(norm_matrix, k=3))
    )

    return matrix


class Callback(CallbackAny2Vec):
    # Evan Parker
    def __init__(self, peitch_to_hc, correlation_matrix: ComputeCorrelation):
        self.epoch_list = []
        self.loss_list = []
        self.correlation = []
        self.peitch_to_hc = peitch_to_hc
        self.hc = []
        self.correlation_matrix = correlation_matrix

    def on_epoch_end(self, model):
        if self.hc == []:
            for key in model.wv.index_to_key:
                self.hc += [self.peitch_to_hc[key]]

        if self.epoch_list == []:
            self.epoch_list += [0]
        else:
            self.epoch_list += [self.epoch_list[-1] + 1]

        self.loss_list += [model.get_latest_training_loss()]

        correlation: float = self.correlation_matrix.compute_correlation(
            cosine_matrix(np.array(
                model.wv.vectors.astype("float64"),
                dtype="float64"
            ))
        )

        self.correlation += [correlation]


def run_model(
    vector_size: int,
    min_count: int,
    window: int,
    sample: int,
    epochs: int,
    corpus: list,
    callback: Callback
) -> tuple:
    """_summary_

    Parameters
    ----------
    vector_size : `int`
        Indicates the size of the embedding vector.
    min_count : `int`
        When words have a frequency inferior to this given value, the words is
        deleted.
    window : `int`
        Size of the window used to take words for the context prediction.
    sample : `int`
        Down sampling words that are to frequent.
    epochs : `int`
        Number of iterations of the neural network.
    corpus : `list`
        It's the corpus... It's actually in the form of `list[list[str]]` where
        the first `list` represents the corpus, the second `list` a sentence,
        and one `str item` a word.

    Returns
    -------
    `tuple`
        Give the last `LOSS` value and the last computed `CORRELATION
        COEFFICIENT`.
    """
    # Build model.
    peitsch2vec = gensim.models.Word2Vec(
        corpus,
        sg=1,
        seed=1,
        workers=cpu_count(),
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        sample=sample
    )

    # Train the the network.
    peitsch2vec.build_vocab(corpus)
    peitsch2vec.train(
        corpus,
        total_examples=len(corpus),
        epochs=epochs,
        callbacks=[callback],
        compute_loss=True
    )

    print(peitsch2vec.wv.index_to_key)
    print(np.array(peitsch2vec.wv.vectors.astype("float64"), dtype="float64").shape)

    return callback.loss_list[-1], callback.correlation[-1]


if __name__ == "__main__":
    model_result: tuple = run_model(
        vector_size=300,
        min_count=2,
        window=1,
        sample=1e-2,
        epochs=50,
        corpus=[list(range(100))] * 100,
        callback=Callback(None, None)
    )

    print(model_result)
