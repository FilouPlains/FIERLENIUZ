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
    def __init__(
        self,
        peitch_to_hc: np.ndarray,
        correlation_matrix: ComputeCorrelation
    ):
        """Instantiate a Callback object.

        Parameters
        ----------
        peitch_to_hc : `np.ndarray`
            To go from Peitsch code to HC.
        correlation_matrix : `ComputeCorrelation`
            An object to compute correlation between a cosine similarity matrix
            and a distance of edition matrix.
        """
        # Fill those next `list` at each end of an epoch?
        self.epoch_list: list = []
        self.loss_list: list = []
        self.correlation: list = []
        self.hc: list = []

        # Set those attributes for later.
        self.peitch_to_hc: np.ndarray = peitch_to_hc
        self.correlation_matrix: np.ndarray = correlation_matrix

    def on_epoch_end(self, model):
        """This method launch itself add the end of a learning iteration.

        Parameters
        ----------
        model : `gensim.models.Word2Vec`
            The word2vec model.
        """
        # Initialize the epoch list.
        if self.epoch_list == []:
            self.epoch_list += [0]
        else:
            self.epoch_list += [self.epoch_list[-1] + 1]

        # Add a loss value to the list.
        self.loss_list += [model.get_latest_training_loss()]

        # To convert Peitsch code into HC.
        filter = np.isin(self.peitch_to_hc[:, 1],
                         np.array(model.wv.index_to_key))

        # Compute the correlation between the cosine similarity matrix and the
        # distance of edition matrix.
        correlation: float = self.correlation_matrix.compute_correlation(
            cosine_matrix(np.array(
                model.wv.vectors.astype("float64"),
                dtype="float64"
            )),
            self.peitch_to_hc[:, 0][filter]
        )

        # Add a correlation to the matrix.
        self.correlation += [correlation]


class WordEmbedding:
    def __init__(
        self,
        vector_size: int,
        min_count: int,
        window: int,
        sample: float,
        epochs: int
    ):
        """Set all the parameters of a word2vec model.

        Parameters
        ----------
        vector_size : `int`
            The size of a embedding vector.
        min_count : `int`
            The frequency limit. If a word have a lower frequency than the
            given value, it will be discard during the learning.
        window : `int`
            The window size to learn the context.
        sample : `float`
            Downsampling very frequent word. If a word is very frequent, this
            will randomly delete some of them for increase the learning
            quality.
        epochs : `int`
            Number of learning iterations.
        """
        self.vector_size: int = vector_size
        self.min_count: int = min_count
        self.window: int = window
        self.sample: float = sample
        self.epochs: int = epochs

    def run(
        self,
        corpus: list,
        peitch_to_hc: np.ndarray,
        correlation_matrix: ComputeCorrelation
    ) -> tuple:
        """Run a word2vec model.

        Parameters
        ----------
        corpus : `list`
            The corpus to learn from.
        peitch_to_hc : `np.ndarray`
            To go from Peitsch code to HC.
        correlation_matrix : `ComputeCorrelation`
            An object to compute correlation between a cosine similarity matrix
            and a distance of edition matrix.

        Returns
        -------
        `tuple`
            Return the last loss and R².

        Raises
        ------
        `RuntimeError`
            Skip an error which occur when the `min_coun` parameter is too
            high.
        """
        # Create the Callback object.
        callback: Callback = Callback(
            peitch_to_hc=peitch_to_hc,
            correlation_matrix=correlation_matrix
        )

        try:
            # Build model.
            peitsch2vec = gensim.models.Word2Vec(
                corpus,
                sg=1,
                seed=1,
                workers=cpu_count(),
                vector_size=int(self.vector_size),
                min_count=int(self.min_count),
                window=int(self.window),
                sample=self.sample
            )
        except RuntimeError as run_time_error:
            # Check the for the next error.
            error: str = ("you must first build vocabulary before training "
                          "the model")

            # This is the attended error? Yes: skip; No: throw the error.
            if error == str(run_time_error):
                return None, None
            else:
                raise RuntimeError(str(run_time_error))

        # Train the the network.
        peitsch2vec.build_vocab(corpus)
        peitsch2vec.train(
            corpus,
            total_examples=len(corpus),
            epochs=int(self.epochs),
            callbacks=[callback],
            compute_loss=True
        )

        return callback.loss_list[-1], callback.correlation[-1]

    def parameter(self) -> np.ndarray:
        """Return all parameters in an `np.ndarray` form.

        Returns
        -------
        `np.ndarray`
            All parameters in an array.
        """
        return np.array([self.vector_size, self.min_count, self.window,
                         self.sample, self.epochs])
