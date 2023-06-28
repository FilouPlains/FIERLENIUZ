"""To create an object to compute correlation between two matrix.
"""


__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "21/04/2023"
__version__ = "1.0.1"
__copyright__ = "CC BY-SA"

# [N]
import numpy as np

# [C]
from correlation_matrix import ComputeCorrelation
# [H]
from hca_out_format_reader import parse_hca_file
# [R]
from random import choice
from running_model import WordEmbedding
# [S]
from sys import path
# [T]
from tqdm import tqdm


class Darwin:
    def __init__(
        self,
        corpus: list,
        peitch_to_hc: np.ndarray,
        corr: ComputeCorrelation
    ):
        self.corpus: list = corpus
        self.peitch_to_hc: np.ndarray = peitch_to_hc
        self.corr: ComputeCorrelation = corr

        self.best_model: WordEmbedding = WordEmbedding(
            vector_size=1,
            min_count=0,
            window=1,
            sample=0,
            epochs=1
        )

        self.model_list: "list[WordEmbedding]" = [self.best_model]
        self.total_model: int = 0

    def __str__(self) -> str:
        to_print: str = ("┏" + "━" * 40 + "┓" + "\n┃ [TOTAL NUMBER OF "
                         f"GENERATED MODEL]: {self.total_model:<4}┃\n┃ [BEST "
                         "MODEL PARAMETERS]:" + " " * 15 + "┃\n")

        param_name: list = ["vector_size", "min_count", "window", "sample",
                            "epochs"]
        space: list = [21, 23, 26, 26, 26]

        for i, param in enumerate(self.best_model.parameter()):
            to_print += f"┃     -{param_name[i]}: {param:<{space[i]}}┃\n"

        return to_print + "┗" + "━" * 40 + "┛"

    def mutate(self, copy: int = 5):
        self.model_list = [self.best_model]

        for i in range(copy):
            vector_size: int = choice([1, -1]) * choice(list(range(500)))

            if vector_size < 1:
                vector_size = 1

            min_count: int = choice([1, -1]) * choice(list(range(500)))

            if min_count <= 0:
                min_count = 0

            window: int = choice([1, -1]) * choice(list(range(500)))

            if window < 1:
                window = 1

            sample: float = choice([1, -1]) * choice([1e-1, 1e-2, 1e-3, 1e-5])

            if sample <= 0:
                sample = 0
            elif sample >= 1:
                sample = 1

            epochs: int = choice([1, -1]) * choice(list(range(500)))

            if epochs < 1:
                epochs = 1

            shift_param: np.ndarray = np.array([
                vector_size,
                min_count,
                window,
                sample,
                epochs
            ])

            self.model_list += [WordEmbedding(
                *shift_param + self.best_model.parameter()
            )]
            self.total_model += 1

    def natural_selection(self):
        loss, r_2 = self.model_list[0].run(
            corpus=self.corpus,
            peitch_to_hc=self.peitch_to_hc,
            correlation_matrix=self.corr
        )

        for model_i in self.model_list[1:]:
            loss_i, r_2_i = model_i.run(
                corpus=self.corpus,
                peitch_to_hc=self.peitch_to_hc,
                correlation_matrix=self.corr
            )

            if loss_i == None or r_2_i == None:
                continue

            if loss_i < loss and r_2_i > r_2:
                loss = loss_i
                r_2 = r_2_i
                self.best_model = model_i

        print(f"[BEST MODEL]: loss={loss:.3f}, r_2={r_2:.3f}")


if __name__ == "__main__":
    corpus, peitch_to_hc = parse_hca_file(
        "/home/lrouaud/Téléchargements/cd-hit/cd-hit_30.out"
    )

    corpus = corpus
    peitch_to_hc = peitch_to_hc

    corr: ComputeCorrelation = ComputeCorrelation(peitch_to_hc[:, 0])

    genetic = Darwin(
        corpus=corpus,
        peitch_to_hc=peitch_to_hc,
        corr=corr
    )

    while True:
        genetic.mutate(copy=15)
        genetic.natural_selection()

        print(genetic)
