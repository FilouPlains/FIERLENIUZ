"""This module contains 3 things:

1) An object `PairewiseContextAnalyzer` to compare pairwise sequences-contexts
using :

    - The divergence between to list, based on `An O(NP) Sequence
      Comparison Algorithm` by Sun WU, Udi MANBER, and Gene MYERS in Information
      Processing Letters, or in the next link:
      https://publications.mpi-cbg.de/Wu_1990_6334.pdf

    - Bray-Curtis distance.

2) An object `MultipleContextAnalyzer` to compare multiple sequences-contexts
using :

    - The dissimilarity percentage.

    - Bray-Curtis distance.

3) A function `center_context` to center `sentences` in `context` around a
`center` object with a distance of `window`. Have to be done to use the
dissimilarity percentage method in `MultipleContextAnalyzer`.
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "10/03/2023"
__version__ = "1.3.0"
__copyright__ = "CC BY-SA"

# [N]
import numpy as np

# [C]
from collections import Counter
# [F]
from functools import reduce
# [S]
from sys import exit as sysexit


class PairewiseContextAnalyzer:
    """To create an object to align two set of objects. 
    """

    def __init__(
        self,
        seq_a: "list | str | np.ndarray",
        seq_b: "list | str | np.ndarray"
    ) -> None:
        """Instantiate an `PairewiseContextAnalyzer` object.

        Parameters
        ----------
        seq_a : `list | str | np.ndarray`
            The first sequence-context to compare.
        seq_b : `list | str | np.ndarray`
            The second sequence-context to compare.
        """
        # Gets sequence-context length.
        self.len_a: int = len(seq_a)
        self.len_b: int = len(seq_b)

        # Invert sequence-context and length so `seq_a` is always the longest
        # one.
        if self.len_a > self.len_b:
            self.len_a, self.len_b = self.len_b, self.len_a
            seq_a, seq_b = seq_b, seq_a

        self.seq_a: str = seq_a
        self.seq_b: str = seq_b

        self.distance: "list[int]" = [None, None]

    def __str__(self) -> str:
        """Change the `print()` function message.

        PRIVATE
        -------

        Returns
        -------
        `str`
            `print()` the distance if it is already compute. Else, `print()` a
            message to say that the distance is not computed. Do it for all
            implemented methods.
        """
        algo_name: "list[str]" = ["O(NP) sequence algorithm",
                                  "Bray-Curtis formula"]
        to_print: str = ""

        # To check if the distance is compute.
        for i, distance in enumerate(self.distance):
            if distance is not None:
                to_print += (f"- Distance compute with {algo_name[i]} is equal "
                             f"to \"{distance:.1f}\".\n")
            else:
                to_print += (f"- Distance compute with {algo_name[i]} was not "
                             "compute.\n")

        return to_print

    def onp_sequence_comparison(self, normalize: bool = True) -> None:
        """The next implemented algorithm is available in `An O(NP) Sequence
        Comparison Algorithm` by Sun WU, Udi MANBER, and Gene MYERS in 
        Information Processing Letters, or in the next link:
        https://publications.mpi-cbg.de/Wu_1990_6334.pdf

        Based on two given `list | np.array | str`, the algorithm computes a the
        difference between those two. It is one the fastest algorithm to do
        this.

        Parameters
        ----------
        normalize : `bool`, optional
            If the data are normalize by both `self.seq_a + self.seq_b`, or do
            we just compute the edition distance. By default, `False`.
        """
        p: int = -1
        # Size different between seq_a and seq_b
        delta: int = self.len_b - self.len_a
        # To shift, because in the original paper, they have a negative indexed
        # array.
        shift: int = self.len_a + 1
        # Create an array full of `-1`, of seq_a and seq_b length + 3.
        fp: list = [-1] * (self.len_a + self.len_b + 3)

        # Check when all operation are done and correct.
        while True:
            p += 1

            for i in range(-p, delta, 1):
                fp[i + shift] = self.__snake(
                    i,
                    max(fp[i - 1 + shift] + 1,
                        fp[i + 1 + shift])
                )

            for i in range(delta + p, delta, -1):
                fp[i + shift] = self.__snake(
                    i,
                    max(fp[i - 1 + shift] + 1,
                        fp[i + 1 + shift])
                )

            fp[delta + shift] = self.__snake(
                delta,
                max(fp[delta - 1 + shift] + 1,
                    fp[delta + 1 + shift])
            )

            if fp[delta + shift] >= self.len_b:
                break

        # Sequence difference and double of p.
        if normalize:
            self.distance[0] = (delta + 2 * p) / (self.len_a + self.len_b)
        else:
            self.distance[0] = (delta + 2 * p)

    def __snake(self, i: int, y: int) -> int:
        """Snake function to follow diagonal edges.

        PRIVATE
        -------

        Parameters
        ----------
        i : `int`
            Array position.
        y : `int`
            `seq_b` position.

        Returns
        -------
        `int`
            The furthest distance between to sequence-context.
        """
        x: int = y - i

        while x < self.len_a and y < self.len_b and \
                self.seq_a[x] == self.seq_b[y]:
            x += 1
            y += 1

        return y

    def bray_curtis(
        self,
        skip_element: "str | int | float" = None,
        do_skip: bool = False
    ) -> None:
        """Bray-Curtis distance, which is 1 - Sørensen-Dice coefficient,
        following the next formula:

        ```
                        2 *|A ∩ B|
        distance = 1 - ————————————
                         |A| + |B|
        ```

        In other word, we compute the distance between two finished `np.array`.
        Note that `|A|` signify the length of a given (computed) set.

        Parameters
        ----------
        skip_element : `str | int | float`
            If this value is set, do not take in consideration this object into
            the intersection. For instance, if we set `skip="-"` and we have an
            intersection equal to `["-", "-", 1]`, we have in result `[1]`. By
            default set to `None`, so we take in consideration all intersection
            objects.

        do_skip : `bool`
            If set to `True`, substrat `1` to the intersection. We do like so
            for the project purpose, as far as we always have one common
            element.
        """
        # To transform a string in a `np.array`.
        if isinstance(self.seq_a, str):
            seq_a: np.array = np.array([*self.seq_a])
        else:
            seq_a: np.array = np.array(self.seq_a)

        # To transform a string in a `np.array`.
        if isinstance(self.seq_b, str):
            seq_b: np.array = np.array([*self.seq_b])
        else:
            seq_b: np.array = np.array(self.seq_b)

        # Compute the intersection.
        intersect: int = np.sum(intersection(self.seq_a, self.seq_b)
                                != skip_element)
        # Compute the coefficient.
        coef: float = 2 * (intersect - do_skip) / (seq_a.shape[0] +
                                                   seq_b.shape[0])

        # Add the coefficient to the list.
        self.distance[1] = 1 - coef

    def compute_distance(self) -> None:
        """An optimal way to compute the distance between two sequences. To do
        so, we check that the two input sequences are 100 % equal. If so, both
        distances are equal to `0.0`. Then we launch `self.bray_curtis`. If this
        distance is equal to `1.0`, this mean that the input sequences are 100 %
        different, and the distance link to `onp_sequence_comparison` is equal
        to the sum of length of both sequences.
        """
        if list(self.seq_a) == list(self.seq_b):
            self.distance = [0.0, 0.0]
            return None

        self.bray_curtis()

        if self.distance[1] == 1.0:
            self.distance[0] = 1.0
        else:
            self.onp_sequence_comparison()


class MultipleContextAnalyzer:
    def __init__(
        self,
        *seq: "list | str | np.ndarray",
        disable_checking: bool = False
    ) -> None:
        """Instantiate an `MultipleContextAnalyzer` object.

        Parameters
        ----------
        *seq : `list | np.ndarray`
            All sequence-context to compare. Not DIRECTLY a list, but much
            something like:

        ```
        >>> MultipleContextAnalyzer(*[["a", "b"], ["c", "d"]])
        ```

            Which going to return something like:

        ```
        ["a", "b"] ["c", "d"]
        ```

            And not:

        ```
        [["a", "b"], ["c", "d"]]
        ```

            SO BEWARE!!!

        disable_checking : `bool`
            Only put `True` when you know and 100 % sure of what you are doing
            and give to this function. Disable the checking of this is list or
            np.array of something that are given, and not something else like a
            string.

            So:

                `True`: no checking, WARNING!

                `False`: checking enable.

            By default, set to `False`.
        """
        self.seq = seq

        if not disable_checking:
            check_str: "list[int]" = list(map(
                lambda item:
                    0 if isinstance(item, list) | isinstance(item, np.ndarray)
                    else 1,
                self.seq
            ))

            if 1 in check_str:
                sysexit("[Err##] Only 'list' or 'np.array' are accepted. "
                        "Please convert the input data.")

        self.distance: "list[int]" = [None, None]

    def __str__(self) -> str:
        """Change the `print()` function message.

        PRIVATE
        -------

        Returns
        -------
        `str`
            `print()` the distance if it is already compute. Else, `print()` a
            message to say that the distance is not computed. Do it for all
            implemented methods.
        """
        algo_name: "list[str]" = ["dissimilarity percentage unaligned",
                                  "Bray-Curtis formula"]
        to_print: str = ""

        # To check if the distance is compute.
        for i, distance in enumerate(self.distance):
            if distance is not None:
                to_print += (f"- Distance compute with {algo_name[i]} is equal "
                             f"to \"{distance:.1f}\".\n")
            else:
                to_print += (f"- Distance compute with {algo_name[i]} was not "
                             "compute.\n")

        return to_print

    def dissimilarity(self):
        """Compute the dissimilarity by doing `1 - identity`.
        """
        # Transform all sequences into a np.array.
        seq: np.array = np.array(self.seq)

        # If all residue of a sequence are identical, the given row will
        # output 0.
        ident: np.array = np.sum(np.absolute(seq[0] - seq[1:]), axis=0)
        # Count the number of 0 (identity) and substract 1.
        dissimilarity: float = 1 - np.sum(ident == 0) / seq.shape[1]

        self.distance[0] = dissimilarity

    def bray_curtis(
        self,
        skip_element: "str | int | float" = None,
        do_skip: bool = False
    ) -> None:
        """Bray-Curtis distance, which is 1 - Sørensen-Dice coefficient,
        following the next formula:

        ```
                         n * |A ∩ B ∩ ... ∩ N|
        distance = 1 - ————————————————————————
                         |A| + |B| + ... + |N|
        ```

        In other word, we compute the distance between `n` finished `np.array`.
        Note that `|A|` signify the length of a given (computed) set.

        Parameters
        ----------
        skip_element : `str | int | float`
            If this value is set, do not take in consideration this object into
            the intersection. For instance, if we set `skip="-"` and we have an
            intersection equal to `["-", "-", 1]`, we have in result `[1]`. By
            default set to `None`, so we take in consideration all intersection
            objects.

        do_skip : `bool`
            If set to `True`, substrat `1` to the intersection. We do like so
            for the project purpose, as far as we always have one common
            element.
        """
        # Compute the intersection.
        intersec: int = np.sum(reduce(intersection, self.seq) != skip_element)

        # Compute the sum of number of items.
        item_sum: int = np.sum(list(map(lambda item: np.array(item).shape[0],
                                        self.seq)))

        # Compute the coefficient.
        coef: float = len(self.seq) * (intersec - do_skip) / item_sum

        # Add the coefficient to the list.
        self.distance[1] = 1 - coef


def center_context(
    context: "np.ndarray",
    window: int,
    center: "str | int | float",
    gap_symbol: "str | int | float" = "-",
    add_gap: bool = True,
    keep_center: bool = True
) -> np.ndarray:
    """Center a set of context. The context have to be in a format like:

    ```
    np.array([[0, 1, 2, 3], [1, 2, 3], [7, 8, 9, 0, 1, 2, 3]])
    ```

    Where the whole `np.array()` is the context, `np.array()[n]` (like
    `[1, 2, 3]`) is a sentence, and `np.array()[n][m]` is a item (like `1`).
    Then, the function return an array centered around `center` for a given
    `window`, like:

    ```
    >>> center_context(
    >>>     context=np.array([[2],[0, 2, 1], [0, 1, 2, 3, 4]]),
    >>>     window=4,
    >>>     center=2
    >>>     gap_symbol="-"
    >>> )

    [["-", 2, "-"], [0, 2, 1], [1, 2, 3]]
    ```

    Parameters
    ----------
    context : `np.ndarray`
        A `np.array` containing a context to center around `center`.

    window : `int`
        How many items to keep around a `center` object. The value indicates
        items from left and right. In other words, if `window=1`, we keep 1 item
        to the left of `center` and 1 item to the right of `center`. If there is
        no item to keep, insert a `gap_symbol`.

    center : `str | int | float`
        The word to center around the context. The words **have to be** in each
        "sentence" of a context.

    gap_symbol : `str | int | float`
        A gap symbol to insert when no item can be kept around `center` for a
        given `window`. By default, "-" is the symbol.

    add_gap: `bool`
        If `True` add gap symbols, else, do not. By default, set to `True`.
    
    keep_center: `bool`
        If `True` keep the word where we center the context around. If we have
        `[1, 2, 3, [4], 5, 6, 7]` (with `[4]` our center word) and we put this
        option to `False`, we will obtain in output `[1, 2, 3, 5, 6, 7]`. By
        default, set to `True`.

    Returns
    -------
    `np.ndarray`
        A context centered around a `center` object.
    """
    # To stock the new formatted context.
    formatted_context: np.ndarray = []

    # Parse all available context and center them.
    for sentence in context:
        # A `center` object have to be in a `sentence`.
        if center not in sentence:
            sysexit(f"[Err#] The object `center` {center} is not present in "
                    f"the `sentence` {sentence}. Please check the input data, "
                    "as `center` has to be in `sentence`.")

        # Convert to a np.array
        sentence: np.ndarray = np.array(sentence)

        # Parse all position where we possibly have a `center` object.
        for i in np.where(sentence == center)[0]:
            # To check to the left that we are not "out of bound". If so, we
            # have to insert gap.
            if i - window < 0:
                left_gap: int = -(i - window) * add_gap
                left_sentence: list = list(sentence[:i])
            else:
                left_gap: int = 0
                left_sentence: list = list(sentence[i - window:i])

            # To check to the right that we are not "out of bound". If so, we
            # have to insert gap.
            if sentence.shape[0] <= i + window + 1:
                right_gap: int = (i + window - sentence.shape[0] + 1) * add_gap
                right_sentence: list = list(sentence[i + 1:])
            else:
                right_gap: int = 0
                right_sentence: list = list(sentence[i + 1:i + window + 1])

            # The new centered and formatted context.
            formatted_context += [left_gap * [gap_symbol] + left_sentence +
                                  [sentence[i]] * keep_center + right_sentence
                                  + right_gap * [gap_symbol]]

    return np.array(formatted_context, dtype=object)


def intersection(
    ar1: "np.ndarray | list",
    ar2: "np.ndarray | list"
) -> np.ndarray:
    """So we can apply the `reduce()` function of `functools` with
    using the `np.intersect1d()` function with:

    ```
    np.intersect1d(
        ar1=...,
        ar2=...,
        assume_unique=True
    )
    ```

    PRIVATE
    -------

    Parameters
    ----------
    ar1 : `np.ndarray | list`
        Input arrays. Will be flattened if not already 1D.
    ar2 : `np.ndarray | list`
        Input arrays. Will be flattened if not already 1D.

    Returns
    -------
    np.ndarray
        1D array of common and **non-unique** elements.
    """
    return np.array(list((Counter(ar1) & Counter(ar2)).elements()))


if __name__ == "__main__":
    print("┏━━━━━━━━━━━━━━━━━━┓")
    print("┃     PAIRWISE     ┃")
    print("┗━━━━━━━━━━━━━━━━━━┛\n")

    Context: object = PairewiseContextAnalyzer("chien", "chien")
    Context.compute_distance()
    print(Context)

    Context: object = PairewiseContextAnalyzer("chien", "niche")
    Context.compute_distance()
    print(Context)

    Context: object = PairewiseContextAnalyzer("abc", "abd")
    Context.compute_distance()
    print(Context)

    Context: object = PairewiseContextAnalyzer(
        [11, 39, 7, 73, 145, 73, 5], [5])
    Context.compute_distance()
    print(Context)

    Context: object = PairewiseContextAnalyzer("abcdefg", "hijklmnop")
    Context.compute_distance()
    print(Context)

    print("┏━━━━━━━━━━━━━━━━━━┓")
    print("┃     MULTIPLE     ┃")
    print("┗━━━━━━━━━━━━━━━━━━┛\n")
    Context: object = MultipleContextAnalyzer([5, 39, 7, 73], [145, 73, 5],
                                              [5])
    Context.bray_curtis()
    print(Context)
    Context: object = MultipleContextAnalyzer([39, 7, 73] * 10, [145, 75] * 10,
                                              [5] * 10)
    Context.bray_curtis()
    print(Context)

    Context: object = MultipleContextAnalyzer([39, 7, 73], [39, 19, 21],
                                              [39, 79, 97], [39, 79, 97])
    Context.bray_curtis()
    Context.dissimilarity()
    print(Context)

    Context: object = MultipleContextAnalyzer([5, 39], [145, 73, 5],
                                              [5])
    Context.bray_curtis()
    print(Context)

    Context: object = MultipleContextAnalyzer(*[[5]] * 100)
    Context.bray_curtis()
    print(Context)

    print("┏━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print("┃     CENTER CONTEXT     ┃")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━┛\n")

    f_context: np.ndarray = center_context(
        context=np.array([[2], [0, 2, 1], [1, 2, 3, 4], [2, 1, 2, 3, 4]],
                         dtype=object),
        window=2,
        center=2,
        add_gap=False
    )

    print("f_context= \n", f_context, "\n")

    f_context: np.ndarray = center_context(
        context=np.array([[2], [0, 2, 1], [1, 2, 3, 4], [2, 1, 2, 3, 4]],
                         dtype=object),
        window=2,
        center=2,
        add_gap=False,
        keep_center=False
    )

    print("f_context= \n", f_context, "\n")

    f_context: np.ndarray = center_context(
        context=np.array([[2, 1], [0, 2, 1], [1, 2, 1, 4], [2, 1, 1, 4]],
                         dtype=object),
        window=2,
        center=2,
        gap_symbol=-1
    )

    print("f_context= \n", f_context, "\n")

    Context: object = MultipleContextAnalyzer(*f_context)
    Context.bray_curtis(skip_element=-1, do_skip=True)
    Context.dissimilarity()
    print(Context)
    
    print("┏━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print("┃     201 : CHECKING     ┃")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━┛\n")

    d1jqna: "list[int]" = [78060471881, 201, 1101, 9, 393, 8791, 1,
                           26265, 9, 6449, 9, 11033, 102677, 11, 83, 17529,
                           58827, 27, 5923, 49, 55595, 6445, 53, 37705, 57,
                           203, 157, 215, 21, 201, 26291, 61, 393, 25, 43,
                           69, 213, 339043, 628403, 9, 73, 29, 7, 19,
                           153, 6327, 117859, 137, 93, 25, 3155, 355, 36497]
    d1izca: "list[int]" = [141, 273, 209, 51, 1001041, 61, 81, 7, 37479, 1,
                           5, 5, 15, 37449, 39, 57993, 201, 1579,
                           51555633]
    d1ujqa: "list[int]" = [17, 89, 39, 5, 674963, 453, 649, 81, 401,
                           15, 657, 7, 38473, 201, 43, 73, 27, 27,
                           13009]
    d3i4ea: "list[int]" = [17, 145, 209, 27, 201, 148709, 35, 913, 5,
                           27, 19635, 2658887, 9, 35, 7, 49, 19, 1093,
                           79, 20753, 53, 81, 77936085713, 77, 9, 4497]


    f_context: np.ndarray = center_context(
        context=np.array([d1jqna, d1izca, d1ujqa, d3i4ea], dtype=object),
        window=10,
        center=201,
        keep_center=False,
        add_gap=False
    )

    print("f_context= \n", f_context, "\n")
    
    d_name: "list[str]" = ["c.1.12.3 - d1jqna", "c.1.12.3 - d1jqna",
                           "c.1.12.5 - d1izca", "c.1.12.7 - d1ujqa",
                           "c.1.12.7 - d3i4ea"]
    summation: int = 0
    count: int = 0

    for i, d_i in enumerate(f_context[:-1]):
        for j, d_j in enumerate(f_context[i + 1:]):
            Context: object = PairewiseContextAnalyzer(d_i, d_j)
            Context.compute_distance()

            print(f">{d_name[i]} vs {d_name[j + 1 + i]}:")
            print(f"{Context.distance[1] * 100:.1f}")
            summation += Context.distance[1]
            count += 1
    
    print(summation)
    print(summation / count * 100)
