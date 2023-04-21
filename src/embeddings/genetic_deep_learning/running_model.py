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
from multiprocessing import cpu_count
# [G]
from gensim.models.callbacks import CallbackAny2Vec


class Callback(CallbackAny2Vec):
    # Evan Parker
    def __init__(self):
        self.epoch_list = []
        self.loss_list = []

    def on_epoch_end(self, model):
        if self.epoch_list == []:
            self.epoch_list += [0]
        else:
            self.epoch_list += [self.epoch_list[-1] + 1]

        self.loss_list += [model.get_latest_training_loss()]

        print(np.array(model.wv.vectors.astype("float64"), dtype="float64"))


def run_model(
    vector_size: int,
    min_count: int,
    window: int,
    sample: int,
    epochs: int,
    corpus: list
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
    callback: Callback = Callback()

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

    return callback.epoch_list[-1], callback.loss_list[-1]


if __name__ == "__main__":
    model_result: tuple = run_model(
        vector_size=300,
        min_count=2,
        window=1,
        sample=1e-2,
        epochs=50,
        corpus=[list(range(100))] * 100
    )

    print(model_result)
