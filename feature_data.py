# feature_data.py

from typing import List
from utils import *
import re
import numpy as np
from collections import Counter
import csv
import numpy as np

class FeatureNorms:
    """
    Indexed list of feature norm objects
    """
    def __init__(self, cue_feature_pairs, normalized=True):
        feature_map = Indexer()
        words = Counter()

        norms = {}

        for pair in cue_feature_pairs:
            word = pair.cue
            words[word] += 1

            if normalized == True:
                feature = pair.translated
                # i think before we were using the untranslated value rather than the translated value...what does this mean? will we see improvement if we retrain?
                value = pair.normalized_translated
            else:
                feature = pair.feature
                value = pair.normalized_feature

            feature_index = feature_map.add_and_get_index(feature)

            if word in norms:
                norms[word][feature_index] = value
            else:
                norms[word] = {feature_index: value}

        self.feature_map = feature_map
        self.feature_norms = norms
        self.length = len(feature_map)
        self.vocab = words

    def get_features(self, word):
        norm = self.feature_norms[word]
        feats = []
        for i in norm:
            feat = self.feature_map.get_object(i)
            feats.append(feat)
        return feats

    def get_feature_vector(self, word):
        norm = self.feature_norms[word]
        vec = np.zeros(self.length)
        for feat, val in norm.items():
            vec[feat] = val
        return vec

    def print_features(self, word):
        norm = self.feature_norms[word]
        pretty = {}
        for feat in norm:
            name = self.feature_map.get_object(feat)
            pretty[name] = norm[feat]
        print("features for %s: %s" % (word, pretty))

    def top_n(self, word, n):
        norm = self.feature_norms[word]
        norm = {k: v for k, v in sorted(norm.items(), key=lambda item: item[1])}
        top_10 = list(norm)[:n]

        feats = []
        for i in top_10:
            feat = self.feature_map.get_object(i)
            feats.append(feat)
        return feats
        #print(top_10)

class FeatureNorm:
    """
    Data wrapper for a single cue word and a list of cue-feature pairs

    We are interested in the untranslated (i.e. lemmatized)
    """
    def __init__(self, word, cue_feature_pairs):
        self.word = word
        features = [feat.feature for pair in cue_feature_pairs]
        values = [feat.normalized_feature for pair in cue_feature_pairs]
        self.features = features
        self.values = values



def read_feature_norms(infile: str): 
    cue_feature_pairs = read_cue_feature_examples(infile)
    norms = FeatureNorms(cue_feature_pairs)

    return norms


class CueFeaturePair:
    """
    Data wrapper for a single cue-target word pair from Buchanan et al. aggregated feature norm dataset
    fields from the csv are
    where,cue,feature,translated,frequency_feature,frequency_translated,n,normalized_feature,normalized_translated,pos_cue,pos_feature,pos_translated,a1,a2,a3,FSG,BSG
    """

    def __init__(self, attributes: dict):
        self.where = attributes["where"]
        self.cue = attributes["cue"]
        self.feature = attributes["feature"]
        self.translated = attributes["translated"]
        self.frequency_feature = float(attributes["frequency_feature"])
        self.frequency_translated = float(attributes["frequency_translated"])
        self.n = attributes["n"]
        self.normalized_feature = float(attributes["normalized_feature"])
        self.normalized_translated = float(attributes["normalized_translated"]) # this is the normalized value of the 'translated' string
        self.pos_cue = attributes["pos_cue"]
        self.pos_feature = attributes["pos_feature"]
        self.pos_translated = attributes["pos_translated"]
        self.a1 = attributes["a1"]
        self.a2 = attributes["a2"]
        self.a3 = attributes["a3"]
        self.fsg = attributes["FSG"]
        self.bsg = attributes["BSG"]

    def __repr__(self):
        return "cue word=" + repr(self.cue) + "; feature=" + repr(self.feature) + "; raw_freq=" + repr(self.frequency_feature)

def read_cue_feature_examples(infile: str) -> List[CueFeaturePair]:
    """
    Reads sentiment examples in the format [0 or 1]<TAB>[raw sentence]; tokenizes and cleans the sentences and forms
    SentimentExamples.

    :param infile: file to read from
    :return: a list of CueTargetPairs parsed from the file
    """
    #fieldnames = ["cue","target","root","raw","affix","cosine2013","jcn","lsa","fsg","bsg"]
    f = open(infile)
    exs = []

    reader = csv.DictReader(f)
    for row in reader:
        #print(row)
        #print(row["CUE"])
        exs.append(CueFeaturePair(row))
    f.close()
    return exs


class SimilarityPair:
    """
    Data wrapper for a single word pair and similarity score.

    Attributes:
        words (List[string]): list of words
        label (int): 0 or 1 (0 = negative, 1 = positive)
    """

    def __init__(self, attributes: dict):
        self.cue = attributes["CUE"]
        self.target = attributes["TARGET"]
        self.root = attributes["root"]
        self.raw = attributes['raw']
        self.affix = attributes['affix']
        self.cosine2013 = attributes['cosine2013']
        self.jcn = attributes['jcn']
        self.lsa = attributes['lsa']
        self.fsg = attributes['fsg']
        self.bsg = attributes['bsg']

    def __repr__(self):
        return "cue word= " + repr(self.cue) + "; target=" + repr(self.target) + "cosine=" + repr(self.raw)

    def __str__(self):
        return self.__repr__()


def read_similarity_examples(infile: str) -> List[SimilarityPair]:
    """
    Reads sentiment examples in the format [0 or 1]<TAB>[raw sentence]; tokenizes and cleans the sentences and forms
    SentimentExamples.

    :param infile: file to read from
    :return: a list of CueTargetPairs parsed from the file
    """
    #fieldnames = ["cue","target","root","raw","affix","cosine2013","jcn","lsa","fsg","bsg"]
    f = open(infile)
    exs = []

    reader = csv.DictReader(f)
    for row in reader:
        #print(row)
        #print(row["CUE"])
        exs.append(SimilarityPair(row))
    f.close()
    return exs

def write_similarity_examples(exs: List[SimilarityPair], outfile: str):
    """
    Writes sentiment examples to an output file with one example per line, the predicted label followed by the example.
    Note that what gets written out is tokenized.
    :param exs: the list of SentimentExamples to write
    :param outfile: out path
    :return: None
    """
    o = open(outfile, 'w')
    for ex in exs:
        o.write(repr(ex.label) + "\t" + " ".join([word for word in ex.words]) + "\n")
    o.close()


def unique_words(exs):
    print("first ten cue_feature examples")
    [ print(ex) for ex in exs[:10] ]
    print()

    unique_words = Indexer()
    for ex in exs:
        unique_words.add_and_get_index(ex.cue)
        unique_words.add_and_get_index(ex.feature)
        unique_words.add_and_get_index(ex.translated)
    return unique_words

if __name__=="__main__":
    # relativize_sentiment_data()
    # exit()
    import sys
    # embs = read_word_embeddings("data/glove.6B.50d-relativized.txt")
    # query_word_1 = sys.argv[1]
    # query_word_2 = sys.argv[2]
    # if embs.word_indexer.index_of(query_word_1) == -1:
    #     print("%s is not in the indexer" % query_word_1)
    # elif embs.word_indexer.index_of(query_word_2) == -1:
    #     print("%s is not in the indexer" % query_word_2)
    # else:
    #     emb1 = embs.get_embedding(query_word_1)
    #     emb2 = embs.get_embedding(query_word_2)
    #     print("cosine similarity of %s and %s: %f" % (query_word_1, query_word_2, np.dot(emb1, emb2)/np.sqrt(np.dot(emb1, emb1) * np.dot(emb2, emb2))))
    exs = read_similarity_examples('data/buchanan/double_words.csv')
    print("first ten similarity examples")
    [ print(ex) for ex in exs[:10] ]
    print()

    exs = read_cue_feature_examples('data/buchanan/cue_feature_words.csv')
    uw = unique_words(exs)
    print(uw)
    print("%s unique words in cue feature data" % len(uw))
