"""Generates embedding vector for a given corpus.
"""

__authors__ = ["DUPRAT Élodie", "ROUAUD Lucas"]
__contact__ = ["elodie.duprat@sorbonne-universite.fr",
               "lucas.rouaud@gmail.com"]
__date__ = "30/01/2023"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


# [A]
import argparse
# [C]
import codecs
# [G]
import glob
# [L]
import logging
# [M]
import multiprocessing
# [O]
import os
# [P]
import pandas as pd
import pickle
# [U]
import umap

# [D]
from datetime import datetime
# [G]
from gensim.models import word2vec as w2v
# [P]
from peitsch_translator import peitsch_translator
# [T]
from tqdm import tqdm


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
        "--window",
        default=5,
        type=int,
        help="window size"
    )

    parser.add_argument(
        "--size",
        default=300,
        type=int,
        help="vector size"
    )

    parser.add_argument(
        "--workers",
        default=multiprocessing.cpu_count(),
        required=False,
        type=int,
        help="number of processes"
    )

    parser.add_argument(
        "--epochs",
        default=5,
        type=int,
        help="number of epochs"
    )

    parser.add_argument(
        "--minTF",
        default=4,
        type=int,
        help="minimum term frequency"
    )

    parser.add_argument(
        "--sample",
        default=1e-3,
        type=int,
        help="down sampling setting for frequent words"
    )

    parser.add_argument(
        "--model",
        required=False,
        type=str,
        help="model file if exists"
    )

    parser.add_argument(
        "-i, --input",
        default="../data/*",
        type=str,
        help="dir to learn from, as a regex for file generation"
    )

    parser.add_argument(
        "-o, --output",
        default="outputs/",
        type=str,
        help="output folder for results"
    )

    parser.add_argument(
        "--alias",
        default="G2V",
        type=str,
        help="model running alias that will be used for model tracking"
    )

    # == OPTIONAL.

    # Transform the input into a dictionary with arguments as key.
    argument = vars(parser.parse_args())
    
    return argument

if __name__ == "__main__":
    # configure logger -
    out_dir = os.path.join(args.output, args.alias)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filename=os.path.join(out_dir, f"{args.alias}.log"), level=logging.INFO)

    # build model
    gene2vec = w2v.Word2Vec(
        sg=1,
        seed=1,
        workers=args.workers,
        vector_size=args.size,
        min_count=args.minTF,
        window=args.window,
        sample=args.sample
    )

    gene2vec.build_vocab(corpus.sentences)
    print("Gene2Vec vocabulary length:", len(gene2vec.wv))
    gene2vec.train(corpus.sentences,
                   total_examples=gene2vec.corpus_count, epochs=args.epochs)
    # save model
    gene2vec.save(os.path.join(
        out_dir, f"{args.alias}_{datetime.today().strftime('%Y-%m-%d')}.w2v"))

    mapper = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=2)
    # train umap
    all_word_vectors_matrix_2d = mapper.fit_transform(
        gene2vec.wv.vectors.astype('float64'))

    # print(gene2vec.wv.key_to_index['K00077.7'])
    # print(all_word_vectors_matrix_2d[gene2vec.wv.key_to_index['K00077.7']])
    # print(gene2vec.wv['K00077.7'], len(gene2vec.wv['K00077.7']))
    # print(gene2vec.wv.distances('K00077.7'))
    # result = gene2vec.wv.similar_by_word('K00077.7')
    # most_similar_key, similarity = result[0]  # look at the first match
    # print(f"{most_similar_key}: {similarity:.4f}")
    # # gene2vec.wv.save('word_vectors.kv')

    points = pd.DataFrame([(word, coords[0], coords[1])
                           for word, coords in [(word, all_word_vectors_matrix_2d[gene2vec.wv.key_to_index[word]])
                                                for word in gene2vec.wv.key_to_index]],
                          columns=["word", "x", "y"])
    with open(os.path.join(out_dir, f"words_umap_{datetime.today().strftime('%Y-%m-%d')}"), 'wb') as o:
        pickle.dump(points, o)
