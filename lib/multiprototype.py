from lib.bert import *
from lib.utils import *
from lib.bnc import *
import os, shutil
from itertools import zip_longest
import numpy as np
import pandas as pd
import csv
import pickle
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
import math

class MultiProtoTypeEmbeddings:
    """
    Wraps an Indexer and a list of 1-D numpy arrays where each position in the list is the vector for the corresponding
    word in the indexer. The 0 vector is returned if an unknown word is queried.
    """
    def __init__(self, word_indexer, vectors, layer, num_clusters):
        self.word_indexer = word_indexer
        self.vectors = vectors
        self.dim = vectors[0].shape[1]
        self.vocab_size = len(word_indexer)
        self.num_prototypes = len(vectors[0])

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_embedding(self, word):
        """
        Returns the embedding for a given word
        :param word: The word to look up
        :return: The UNK vector if the word is not in the Indexer or the vector otherwise
        """
        word_idx = self.word_indexer.index_of(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.index_of("UNK")]

    def get_embeddinge_at_absolute_index(self, index):
        None


    def find_nearest_neighbor(self, query_vector):
        # we have to find 
        query_vec = np.array(query_vector)
        #print(self.vectors.shape)

        num_tokens = self.vectors.shape[0]
        num_prototypes = self.num_prototypes
        num_dims = self.dim

        sqz = self.vectors.reshape((num_tokens * num_prototypes, num_dims))
        #print(sqz.shape)
        #print(query_vec.shape)
        sims = cosine_similarity_matrix(sqz, query_vec)

        #print(sims.shape)
        # TODO this might return a tie.... what then?
        max_value = max(sims)
        #print(max_value)

        neighbor_index = np.where(sims == max_value)
        neighbor_index = neighbor_index[0][0]
        #print(neighbor_index)
        type_index = int(neighbor_index / self.num_prototypes)
        prototype_index = int(math.remainder(neighbor_index, self.num_prototypes))
        #print(type_index)
        #print(prototype_index)
        neighbor_vec = self.vectors[type_index][prototype_index]
        neighbor_label = self.word_indexer.get_object(type_index) + '_' +str(prototype_index)
        #print(neighbor_label)
        return neighbor_label, neighbor_vec

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def parseline(line: str):
    """
    reads in a line from embedding file.
    returns a tuple containing a word and a numpy vector
    """
    if line.strip() != "":
        space_idx = line.find(' ')
        word = line[:space_idx]
        numbers = line[space_idx+1:]
        float_numbers = [float(number_str) for number_str in numbers.split()]
        vector = np.array(float_numbers)
        return(word, vector)
    else:
        return None

def read_multiprototype_embeddings(embeddings_file: str, layer=8, num_clusters=1) -> MultiProtoTypeEmbeddings:
    """
    Should return an embedding for that particular parameter specification.
    This is loaded from the textfile representation, or else generated from the data we have word by word

    Loads the given embeddings (ASCII-formatted) into a WordEmbeddings object. Augments this with an UNK embedding
    that is the 0 vector. Reads in all embeddings with no filtering -- you should only use this for relativized
    word embedding files.
    :param embeddings_file: path to the file containing embeddings
    :return: WordEmbeddings object reflecting the words and their embeddings
    """

    f = open(embeddings_file)
    word_indexer = Indexer()
    embeddings = []

    n = num_clusters

    for lines in grouper(f, n, ''):
        assert len(lines) == n
        # process N lines here
        prototypes = []
        for line in lines:
            word, vector = parseline(line)
            #print(word)
            #print(vector.shape)
            prototypes.append(vector)

        word_indexer.add_and_get_index(word)
        embeddings.append(np.array(prototypes))

    f.close()
    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(len(embeddings[0])) + " X " + repr(embeddings[0][0].shape[0]))

    # Turn vectors into a n-D numpy array
    return MultiProtoTypeEmbeddings(word_indexer, np.array(embeddings), layer, num_clusters)




def generate_clusters(bert, embeddings_dir, words, layers, cluster_sizes):
    i = 0
    for i in range(0, len(words)):
        word = words.get_object(i)

        word_results = []

        i+=1
        if i % 20 == 0:
            print("processed %s words" % i)
            print("calculating clusters for %s" % word)

        # if you have the pickle file already, continue
        outpath = os.path.join(embeddings_dir, word, 'analysis_results', 'clusters.p')
        if os.path.isfile(outpath):
           print("already have clusters for %s" % word)
           continue
        else:
            print("calculating clusters for %s" % word)


        # create a directory to store all our clustering results in
        results_dir = os.path.join(embeddings_dir, word, 'analysis_results')    
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)

        # read in tokens for this word
        data = read_tokens_for(word)
        #print(data)


        outpath = os.path.join(results_dir, 'clusters.p')


        if data:
            # get the model activation for each token
            for token in data.tokens:
                token.vector = bert.get_bert_vectors_for(word, token.sentence)   
            # get rid of the data points we weren't able to vectorize (due to length)
            tokens = list(filter(lambda tok: tok.vector != None, data.tokens))

            # gwt the clusters!
            for layer in layers:
                #print("layer %s" % layer)

                for k in cluster_sizes:
                    #print("clusters %s" % k)

                    sub_results = calculate_clusters_for(word, tokens, layer, k, bert)
                
                    word_results.append(sub_results)

            pickle.dump(word_results, open(outpath, 'wb'))
        else:
            print("no tokens collected for %s" % word)

def calculate_clusters_for(word, tokens, layer, k, bert):
    if (len(tokens)) >= k:

        layer_vectors = [ token.vector[layer] for token in tokens]
        clusters = []

        # calculate clusters
        kmeans_obj = KMeans(n_clusters=k)
        kmeans_obj.fit(layer_vectors)
        label_list = kmeans_obj.labels_
        cluster_centroids = kmeans_obj.cluster_centers_


        datapoints = []
        # store cluster_id with token
        for index,token in enumerate(tokens):
            datapoint = {}
            datapoint['word'] = word
            datapoint['sentence'] = token.sentence
            datapoint['uid'] = token.uid
            datapoint['cluster_id'] = label_list[index]

            datapoints.append(datapoint)

        # retrieve centroid for each cluster and uids of sentences in cluster:
        for cluster_index in range(k):
            sentence_uids = []
            cluster_vectors = []

            for index, datapoint in enumerate(datapoints):
                if datapoint['cluster_id'] == cluster_index:
                    sentence_uids.append(datapoint['uid'])
                    cluster_vectors.append(layer_vectors[index])

            single_cluster_data = {'word': word,
                        'layer': layer,
                        'k_clusters': k,
                        'cluster_id': cluster_index,
                        'centroid': cluster_centroids[cluster_index],
                        'sentence_uids': sentence_uids,
                        }
            clusters.append(single_cluster_data)      
        return clusters
    else:
        return None

def read_clusters(embeddings_dir, word):   
    try:
        cluster_path = os.path.join(embeddings_dir, word, 'analysis_results', 'clusters.p')
        """
             this is a list of dicts with the structure:
                            {'word': tokens[0]['word'],
                            'layer': layer,
                            'k_clusters': k,
                            'cluster_id': cluster_index,
                            'centroid': cluster_centroids[cluster_index],
                            'sentence_uids': sentence_uids,
                            'within_cluster_variance': cluster_variance
                            }
        """
        data = pickle.load(open(cluster_path, 'rb'))

        # the 'if item' removes None values for cluster sizes we didnt have enough tokens for
        data = [item for sublist in data if sublist for item in sublist]
        
        columns = ['word', 'layer', 'k_clusters', 'cluster_id', 'centroid', 'sentence_uids', 'within_cluster_variance', 'average_pairwise_token_distance']
        df = pd.DataFrame.from_records(data, columns=columns)
        return df
    except:
        return None
    


def read_centroids_for_word_at_layer_and_cluster(embeddings_dir, word, layer_number, k):
    df = read_clusters(embeddings_dir, word)
    if df is None:
        print("no tokens collected for %s" % word)
        return None
    df = df[df['layer'] == layer_number]
    df = df[df['k_clusters'] == k]
    word_centroids = df['centroid']
    if len(word_centroids) > 0:
        return word_centroids
    else:
        return None

def prepare_embeddings(embeddings_dir, outfile, unique_words, layer, k_clusters) -> MultiProtoTypeEmbeddings:
    o = open(outfile, 'w')

    for word in unique_words:
        data = read_centroids_for_word_at_layer_and_cluster(embeddings_dir, word, layer, k_clusters)
        if data is None:
            continue
        for row in data:
            number_str = np.array2string(row, precision=8, max_line_width=None)
            number_str = number_str[1:-1].replace("\n", "") # get rid of brackets
            emb_str = word + ' ' + number_str + "\n"
            o.write(emb_str)
    return None

