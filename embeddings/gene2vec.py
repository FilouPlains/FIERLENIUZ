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
# [T]
from tqdm import tqdm


class Corpus(object):
    def __init__(self, dir_path):
        self.dirs = dir_path
        self.len = None
        self.corpus = None
        self.sentences = None
        self.token_count = None

    def load_corpus(self):
        # initialize rawunicode , all text goes here
        corpus_raw = u""
        files = glob.glob(self.dirs)
        files.sort()

        for f in tqdm(files):
            with codecs.open(f, "r", "utf-8") as book_file:
                corpus_raw += book_file.read()
        # set current corpus
        self.corpus = corpus_raw

    def make_sentences(self, delim=". "):
        # create sentences from corpus
        if self.corpus == None:
            print(
                "Error: no corpus object found, use load_corpus function to generate corpus object")
            return
        raw_sentences = self.corpus.split(delim)

        sentences = []

        for raw_sentence in tqdm(raw_sentences):
            if len(raw_sentence) > 0:
                sentences.append(raw_sentence.split())

        self.sentences = sentences

        print(f"{len(self.sentences)=}")

        # update number of tokens in corpus
        self.token_count = sum([len(sentence) for sentence in sentences])

        print(f"{self.token_count=}")


def main(args):
    # configure logger -
    out_dir = os.path.join(args.output, args.alias)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filename=os.path.join(out_dir, f"{args.alias}.log"), level=logging.INFO)
    corpus = Corpus(args.input)

    print("corpus.load_corpus()")
    corpus.load_corpus()
    corpus.make_sentences()
    # Seed for the RNG, to make the results reproducible.
    seed = 1
    if args.workers == None:
        args.workers = multiprocessing.cpu_count()
    # build model
    gene2vec = w2v.Word2Vec(
        sg=1,
        seed=seed,
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


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--window', default=5, type=int, help='window size')
    argparse.add_argument('--size', default=300, type=int, help='vector size')
    argparse.add_argument('--workers', required=False,
                          type=int, help='number of processes')
    argparse.add_argument('--epochs', default=5,
                          type=int, help='number of epochs')
    argparse.add_argument('--minTF', default=4, type=int,
                          help='minimum term frequency')
    argparse.add_argument('--sample', default=1e-3, type=int,
                          help='down sampling setting for frequent words')
    argparse.add_argument('--model', required=False,
                          type=str, help='model file if exists')
    argparse.add_argument('--input', default='../data/*', type=str,
                          help='dir to learn from, as a regex for file generation')
    argparse.add_argument('--output', default='outputs/',
                          type=str, help='output folder for results')
    argparse.add_argument('--alias', default='G2V', type=str,
                          help='model running alias that will be used for model tracking')
    params = argparse.parse_args()

    main(params)
