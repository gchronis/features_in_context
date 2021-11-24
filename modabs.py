from time import strftime

__author__ = 'Alex'

import numpy as np

"""
added by gabriella
"""
from multiprototype import *
from feature_data import *
from utils import cosine_similarity_matrix
from models import FeatureClassifier
from scipy import spatial
import math


class MAD(FeatureClassifier):
    def __init__(self, beta=4.0, mu1=1, mu2=1, mu3=None, mu4=None, tol=1e-6, NNk=5, NN_flag=None, alpha_type="decay", use_pycuda=False):
        self.G = None
        self.labeled_nodes = None
        self.beta = beta
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.mu4 = mu4
        self.tol = tol
        self.NNk = NNk
        self.NN_flag = NN_flag
        self.alpha_type = alpha_type
        self.use_pycuda = use_pycuda
        self.predictions = None
        self.word_indexer = None
        self.num_prototypes = None
        self.feature_norms = None
        self.multipro_embeddings = None
        self.kd_tree = None
    

    def _entropy(self, P):
        P_fix = np.copy(P)
        P_fix[np.where(P_fix <= 0.0)] = 1.0
        return (-1*P_fix*np.log(P_fix)).sum(axis=1)

    def _f(self, x):
        return np.log(self.beta)/np.log(self.beta + np.exp(x))

    def _cvdv(self, Hv):
        self._cv = self._f(Hv)
        self._dv = (1.0-self._cv)*np.sqrt(Hv)*self._label_array
        self._zv = self._cv + self._dv
        self._zv[np.where(self._zv < 1.0)] = 1.0
        self._p_cont = self._cv/self._zv
        self._p_inj = self._dv/self._zv
        self._p_abdn = 1 - self._p_cont - self._p_inj

    def _k_NN(self, X, k=10):
        X = X - np.diag(np.diag(X))
        num_rows, num_columns = X.shape
        new_X = np.zeros((num_rows, num_columns), dtype=X.dtype)
        index_mat = np.argpartition(X, -k)
        index_mat = index_mat[:,-k:]
        for i in range(0, num_rows):
            new_X[i, index_mat[i, :]] = 1.0
        return new_X


    def _softmax(self, X, c=1):
        P = np.exp(c*X)
        P = P/np.sum(P, axis=1)
        return P

    def _powmax(self, X, c=1):
        P = X.astype(np.float32)
        P = P**c
        P = P/np.sum(P, axis=1)
        return P

    def _reweigh_X(self, X):
        if self.NN_flag:
            num_drops = self.NNk
            X = self._k_NN(X, num_drops).T
        else:
            num_drops = self.NNk
            if self.alpha_type == "decay":
                alphas = [1/float(2**n) for n in range(0, num_drops)]
            elif self.alpha_type == "even":
                alphas = [1 for _ in range(0, num_drops)]
            else:
                alphas = None
                print("Illegal alpha_type given")
                exit(1)
            sum_alphas = sum(alphas)
            alphas = [x/float(sum_alphas) for x in alphas]
            ks = [1] + [5*(2**n) for n in range(0, num_drops-1)]
            comb_X = np.zeros(X.shape)
            for alpha, k in zip(alphas, ks):
                comb_X += alpha*self._k_NN(X, k).T
            X = comb_X
        bottom = X.sum(axis=1)
        bottom[np.where(bottom==0)] = 1.0
        P = X/bottom[:,np.newaxis]
        self._cvdv(self._entropy(P))
        return self._p_cont[:, np.newaxis] * X

    def _reweigh_X_bare(self, X):
        P = X/X.sum(axis=1, keepdims=True)
        self._p_cont = np.ones(self._num_samples)
        self._p_inj = np.ones(self._num_samples)
        self._p_abdn = np.zeros(self._num_samples)
        return P

    def _find_nearest_neighbor(self, vector):
        vector = np.array(vector)
        vector = np.expand_dims(vector, axis=1)
        print("calculating similarities with neighbors")
        print(self.predictions.shape)
        print(vector.shape)
        sims = cosine_similarity_matrix(self.predictions, vector)
        print("done calculating similarities with neighbors")

        max_value = max(sims)
        max_index = sims.index(max_value)
        neighbor = self.predictions[max_index]
        # try:
        #     d, i = self.kd_tree.query(vector, 1) # 1 is number of neighbors to return
        # except:
        #     print("building kdtree for nneighbor queries for modabs model")
        #     self.kd_tree = spatial.KDTree(self.predictions)
        #     print("done building kdtree for nneighbor queries for modabs model")
        #     d, i = self.kd_tree.query(vector, 1)
        # vector = self.kd_tree.data[i]

        #word_index = i - (i % self.num_prototypes)

        # have to do this in a fucked up reverse lookup because you didnt make a nice indexer
        #rev_index = [w for w in self.predictions if word_index[w] == i]
        #logits = rev_index[0]

        return neighbor

    def fit(self, X, Y, C=None):
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        if not self.use_pycuda:
            matmult = lambda A,B: A.dot(B)
            matadddot = lambda A,B,C: A.dot(B) + C
            elemult = lambda A,B: A*B
        else:
            import mult as CUDA_mult
            import imp
            matmult = lambda A,B: CUDA_mult.cuda_dot(A, B)
            matadddot = lambda A,B,C: CUDA_mult.cuda_add_dot(A, B, C)
            elemult = lambda A,B: CUDA_mult.cuda_mult(A, B)
        self._num_samples, self._num_labels = Y.shape
        self._label_array = np.zeros(self._num_samples)
        self._label_array[np.where(Y.any(axis=1))] = 1.0
        self._orig_Y = np.zeros((self._num_samples, self._num_labels + 1))
        self._orig_Y[:,:-1] = Y

        X = self._reweigh_X(X)
        X = X.astype(np.float32)
        D = X + X.T


        regularizer = self.mu1*(self._p_inj[:,np.newaxis]*self._orig_Y)
        antagonizer = self._p_abdn*np.full(self._num_samples, self.mu3)
        regularizer[:,self._num_labels] = antagonizer

        M = self.mu1*self._p_inj*self._label_array + self.mu2*(D.sum(axis=1) - np.diag(D)) + self.mu3
        if C is not None:
            C = C.astype(np.float32)
            C_expand = np.zeros((self._num_labels + 1, self._num_labels + 1), dtype=np.float32)
            C_expand[:-1,:-1] = C
            M = np.tile(M, (self._num_labels + 1, 1)).T + self.mu4*(np.tile(C_expand.sum(axis=1), (self._num_samples, 1)))
            M_inv = 1.0/M
        else:
            M = np.tile(M, (self._num_labels + 1, 1)).T
            M_inv = 1.0/M
        part1 = M_inv*regularizer
        part2 = self.mu2*M_inv
        if C is not None:
            part3 = self.mu4*M_inv
        old_Y = self._orig_Y - 2*self.tol
        new_Y = self._orig_Y
        step=0
        diff = np.linalg.norm(old_Y-new_Y)
        changes = []
        while (diff > self.tol):
            print("inner step", step, np.linalg.norm(old_Y-new_Y), strftime("%Y-%m-%d %H:%M:%S"))

            old_Y = np.copy(new_Y)
            old_Y = old_Y.astype(np.float32)
            new_Y = M_inv*(regularizer + self.mu2*(matmult(D, old_Y)))
            if C is not None:
                new_Y = new_Y + part3*matmult(old_Y,C_expand)
            diff = np.linalg.norm(old_Y-new_Y)
            changes = [diff] + changes
            changes = changes[:6]
            if len(changes) == 6 and changes[0] >= changes[1]:
                break
            step += 1
            if step >= 100:
                break

        new_Y = np.delete(new_Y, self._num_labels, 1)
        return new_Y

    def test(self):
        a = np.random.random((4,4))
        print(a)
        print(self._k_NN(a, 2))
        #exit()

    def predict(self, word: str):
        labels = [word + '_' + str(i) for i in range(0, self.num_prototypes)]
        #print(labels)
        vectors = np.empty([self.num_prototypes, self.output_dims])
        #print(vectors.shape)
        for i in range(0,len(labels)):
            label = labels[i]
            #print(label)
            #print(self.word_indexer.objs_to_ints)
            word_index = self.word_indexer.index_of(label)
            #print(word_index)
            vec = self.predictions[word_index]
            #print(vec.shape)

            vectors[i,:] = vec
        #vectors = self.predictions[word_index:(word_index + self.num_prototypes), :]
        #print(vectors.shape)
        avg = np.average(vectors, axis=0)
        return avg

    #def predict_top_n_features(self, word: str, n: int, vec=None):
    #    raise Exception("Not implemented")



    def predict_in_context(self, word, sentence, bert):
        # generate bert vector for word
        vec = bert.get_bert_vectors_for(word, sentence)
        # get the layer we care about
        vec = vec[8]

        # reshape to be vertical
        vec = vec.reshape(1, -1)
        #print("after reshape")
        #print(vec.shape)


        """
        find the nearest neighbor to the input vector
        """
        # TODO to predict the nearest neighbor by word we have to fix the counter to an indexer and retrain all the saved models =/

        # find the closest input embedding to our context vector
        label, vec = self.multipro_embeddings.find_nearest_neighbor(vec)

        # use the label to find the projection we have stored for that embedding
        index = self.word_indexer.index_of(label)

        logits = self.predictions[index]

        return logits


def train_mad(train_exs: List[str], dev_exs: List[str], test_exs: List[str], multipro_embs: MultiProtoTypeEmbeddings, feature_norms: FeatureNorms, args) -> MAD:

    """
    First prepare a dataset with our labeled examples
    (all of the instances , labeled with the features)
    """

    # for each of the train examples, make an instance of all the prototypes and label them with the gold vector


    #initialize empty np array to hold  data
    # [number of vectors x size of vectors]
        # we have k times the number of training words, because we have multi-prototype clusters
        # add to that k times the number of dev words, and 
    num_samples = (len(train_exs) +len(dev_exs) + len(test_exs))     * multipro_embs.num_prototypes
    X = np.empty([num_samples, multipro_embs.dim])
    # initialize empty matrix to hold gold labels
    # [number of vectors x size of feature norm fectors]
    Y = np.empty([num_samples, feature_norms.length])

    word_indexer = Indexer()

    # iterate through the words
    i=0
    for word in train_exs:

        # look up the embedding
        emb = multipro_embs.get_embedding(word)
        # look up the feature norm
        norm = feature_norms.get_feature_vector(word)

        # iterate through the prototypes for each word
        #print(emb.shape)        
        for index in range(0, emb.shape[0]):
            # this adds something to the index like 'apple_3' so we know what the prediction is for each thing
            word_indexer.add_and_get_index(word + '_' + str(index))


            # add vector to training xs
            vector = emb[index, :]
            #print(vector.shape)
            X[i] = vector

            # add feature norm to training ys
            #print(norm.shape)
            Y[i] = norm

            i+=1

    for word in (dev_exs + test_exs):
        # look up the embedding
        emb = multipro_embs.get_embedding(word)
        # look up the feature norm
        norm = feature_norms.get_feature_vector(word)

        for index in range(0, emb.shape[0]):
            # this adds something to the index like 'apple_3' so we know what the prediction is for each thing
            word_indexer.add_and_get_index(word + '_' + str(index))

            vector = emb[index,:]
            X[i] = vector

            # make these words unlabeled
            Y[i,:] = 0.0

            i+=1

    print("training length")
    print(X.shape)
    print(Y.shape)

    C=None


    """
    calculte similarity matrix for input...this is going to take a while and be BIG
    """
    # similarity_matrix is a square matrix with similarities
    # e.g., similarity_matrix[i,j] = cosine(word[i], word[j])
    similarity_matrix = cosine_similarity_matrix(X,X)

    print(similarity_matrix.shape)

    model = MAD(beta=4, mu1=args.mu1, mu2=args.mu2, mu3=args.mu3, mu4=args.mu4, NNk=args.nnk)

    #print(len(X))
    #print(len(this_feature_for_all_words))

    #predictions = medal.fit(similarity_matrix, properties, C)

    # this is a matrix of all the predicted feature vectors for each token in the sample
    predictions = model.fit(similarity_matrix, Y, C)
    model.predictions = predictions
    model.word_indexer = word_indexer # maps predictions to the word they predict and prototype
    model.num_prototypes = multipro_embs.num_prototypes
    model.feature_norms = feature_norms
    model.multipro_embeddings = multipro_embs
    model.output_dims = predictions.shape[1]
    
    """
    we also need i think our unlabeled examples?
    """


    #print("=======EVAL ON TRAIN SET=======")
    #evaluate(model, train_exs, feature_norms, args, debug='false')
    #print("=======EVAL ON DEV SET=======")
    #evaluate(model, dev_exs, feature_norms, args, debug='info')

    return model


if __name__ == '__main__':
    medal = MAD(beta=4, mu1=1, mu2=1, mu3=1, mu4=1, NNk=2)


    medal.test()
    """
    load up multi-pro embeddings
    and caluclate similarity matrix
    """
    #layer = 0
    #clusters = 5
    #embedding_file = './data/multipro_embeddings/layer'+ str(layer) + 'clusters' + str(clusters) + '.txt'
    #embs = read_multiprototype_embeddings(embedding_file, layer=args.layer, num_clusters=args.clusters)

    #feature_norms = McRaeFeatureNorms('data/mcrae/CONCS_FEATS_concstats_brm/concepts_features-Table1.csv')

    """
    label each one with the properties for that lemma
    """

    """
    shuffle them???
    but most important---unlabel the last few. 
    """

    """
    then FIT
    """

    # similarity_matrix is a square matrix with similarities
    # e.g., similarity_matrix[i,j] = cosine(word[i], word[j])
    similarity_matrix = np.random.random((20, 20))
    # properties are your gold values
    # e.g., properties[i,j] = the value of property j for word i
    properties = np.random.random((20, 3))
    # put zero values for unlabeled words/words you want to label
    # below I made the last 5 words be unlabeled
    properties[15:, :] = 0.0
    # I forget what C is for, but hopefully it doesn't matter **shrug**
    # C = np.random.random((20,20))
    C=None


    predictions = medal.fit(similarity_matrix, properties, C)
    print("model predictions")
    print(predictions)
