"""
Script for building the required vocabulary

Arguments:
--recipe1M:     boolean variable that specifies whether the vocabulary should
                be built from the plain annotations of the YouCook2 dataset or the Recipe1M dataset.
                True for  using the recipe1M dataset
--threshold     minimum amount of word counts for a word to be incorporated into the vocabulary.
                The default is set to zero
"""


import nltk
import pickle
import argparse
import numpy as np
from collections import Counter
from gensim.models import KeyedVectors
from tqdm import tqdm
import pandas as pd
import sys

# specify the root path of the project for importing own modules
sys.path.append("C:/Users/User/foodcap")

# misc
import src.config as config
from src.utils import load_json_data


class Vocabulary(object):
    '''
    Class that creates a vocabulary object that contains a dictionary that maps each word of the vocabulary to its
    numerical index and a second dictionary that holds the inverse relation. For the standard case the vocabulary is
    built from the pain annotations of the YouCook2 dataset.
    If the vocabulary is built from the Recipe1M dataset, the vocabulary model of this dataset is loaded and used as
    basis for the vocabulary. Additionally a weight matrix that stores the similarities of words in the vocabulary
    is created.
    '''
    def __init__(self, data_dir, recipe1M = True):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.recipe1M = recipe1M
        if self.recipe1M:
            self.vocab_model = KeyedVectors.load_word2vec_format(data_dir+config.DATA["recipe1M_model"],
                                                                 binary=True)
            self.weight_matrix = self.vocab_model.vectors
        else:
            self.weight_matrix = []

    def add_vocab_from_Recipe1M(self):
        """
        Function for adding the words from the Recipe1M dataset to the vocabulary.
        The lookup dictionaries of the vocabulary object are filled accordingly
        :return:
        """
        if self.recipe1M:
            for word in tqdm(self.vocab_model.index2word):
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def add_word(self, word):
        """
        Function for adding a specific word to the vocabulary.
        The lookup dictionaries of the vocabulary object are filled accordingly.

        :param word: word that should be added to the vocabulary
        :return:
        """
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word

            # If the vocabulary is built from the Recipe1M dataset the weight matrix is extended with a new row
            # of random weights as for words that are not present in the Recipe1M dataset there does not exist a clear
            # similarity relationship to the words that are present

            if self.recipe1M:
                self.weight_matrix = np.vstack([self.weight_matrix, np.random.normal(scale=0.6, size=(300,))])

            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
    
    def __str__(self): 
        return str(self.word2idx), str(self.idx2word)


def build_vocab(threshold, recipe1M, data_dir):
    """
    Function for building the vocabulary. Either only on the basis of the plain annotations of the YouCook2 dataset or
    with all the words from the Recipe1M dataset.
    :param threshold:   minimum amount of word counts for a word to be incorporated into the vocabulary.
    :param recipe1M:    boolean variable that specifies whether the vocabulary should
                        be built from the plain annotations of the YouCook2 dataset or the Recipe1M dataset.
                        True for  using the recipe1M dataset
    :param data_dir:    path to the data directory of the project
    :return: vocabulary object
    """
    # initialize the vocabulary
    vocab = Vocabulary(data_dir, recipe1M)

    # add the words from recipe1M  to the vocabulary if recipe1M is true
    if recipe1M:
        print("Add vocab from recipe1M")
    vocab.add_vocab_from_Recipe1M()

    # add all the  words from the annotations of the YouCook2 dataset to the vocabulary
    print("add all words from YouCook2 annotations")
    annotations_path = data_dir + config.DATA["annotations_file"]
    data = load_json_data(annotations_path)
    # flatten the annotations to each segment, df_annotations.index matches df_data.index
    yc2_annotations = data.annotations.apply(pd.Series).stack().reset_index(level=1, drop=True).to_frame('annotations')
    yc2_annotations = yc2_annotations.annotations.apply(pd.Series)

    counter = Counter()
    print("Tokenize...")
    for idx, caption in tqdm(enumerate(yc2_annotations['sentence'])):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        
        if (idx+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(idx+1, len(data)))
            
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # add some special tokens to the vocabulary
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    
    # Add the words to the vocabulary.
    for i, word in tqdm(enumerate(words)):
        vocab.add_word(word)
    return vocab


def main(args):
    """
    Function that executes the pipeline for building the vocabulary based on the given arguments
    :param args:    arguments that specify from which source the vocabulary should be built from and how often a word
                    should at least appear in the annotations to be added to the vocabulary
    :return:
    """

    data_dir = config.DATA["data_dir"]

    # build the vocabulary
    vocab = build_vocab(threshold=args.threshold, recipe1M=args.recipe1M, data_dir=data_dir)

    # save the vocab in the data directionary
    if args.recipe1M:
        vocab_path = data_dir + "vocab_recipe1M.pkl"
    else:
        vocab_path = data_dir + "vocab.pkl"
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--threshold', type=int, default=0,
                        help='minimum word count threshold')
    parser.add_argument('--recipe1M', dest='recipe1M', action='store_true')
    parser.set_defaults(recipe1M=False)
    args = parser.parse_args()
    main(args)





