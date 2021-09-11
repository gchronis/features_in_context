from time import strftime

__author__ = 'Alex'

import numpy as np

"""
added by gabriella
"""
from multiprototype import *
from feature_data import *

class MAD:
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
                print "Illegal alpha_type given"
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
            print "inner step", step, np.linalg.norm(old_Y-new_Y), strftime("%Y-%m-%d %H:%M:%S")

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
        print a
        print self._k_NN(a, 2)
        exit()


if __name__ == '__main__':
    medal = MAD(beta=4, mu1=1, mu2=1, mu3=1, mu4=1, NNk=2)


    """
    load up multi-pro embeddings
    and caluclate similarity matrix
    """


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
    print predictions
