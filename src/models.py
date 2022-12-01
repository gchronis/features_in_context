# models.py
import numpy as np
from src.utils import *
from src.multiprototype import *
from src.feature_data import *
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error


from typing import List

def add_models_args(parser):
    """
    Command-line arguments to the system related to your model.  Feel free to extend here.  
    """
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=None, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--hidden_size', type=int, default=None, help='size of hidden state')
    parser.add_argument('--dropout', type=float, default=None, help='dropout probability')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes
    #parser.add_argument('--embedding_size', type=int, default=50, help='size of embedding to train')
    #parser.add_argument('--pretrained', type=bool, default=True, help='Boolean indicating whether to start with pretrained vectors')
    """
    PLSR args
    """
    parser.add_argument('--plsr_n_components', type=int, default=None, help='number of dimensionality reduction components to keep')
    parser.add_argument('--plsr_max_iter', type=int, default=None, help='The maximum number of iterations of the power method when algorithm=nipals. Ignored otherwise.')

    """
    ModAbs args
    """
    parser.add_argument('--mu1', type=float, default=None, help='mu_inj (Talukdar and Kramer 2009)')
    parser.add_argument('--mu2', type=float, default=None, help='mu_cont (Talukdar and Kramer 2009)')
    parser.add_argument('--mu3', type=float, default=None, help='mu_abdn (Talukdar and Kramer 2009)')
    parser.add_argument('--mu4', type=float, default=None, help='ModAds NNk (Rosenfeld and Erk 2019)')
    parser.add_argument('--nnk', type=int, default=None, help='ModAds equal/decay n (Rosenfeld and Erk 2019')


class FeatureClassifier(object):
#     """
#     Classifier to classify predict a distribution over features for a bert word-type embedding
#     """

    def __init__(self, nn, vectors, feature_norms):
        self.nn = nn
        self.word_vectors = vectors
        self.feature_norms = feature_norms


    def predict(self, word: str):
        """
        Makes feature predictions for the word  given by word
        :param word:
        :return: logits over all output classes (output sized vector)
        """

        x = form_input(word, self.word_vectors)
        logits = self.nn.forward(x)
        logits = logits.detach().numpy()

        # these are in shape [k x n]  where k is the number of prototypes
        # you need to choose somehow which embedding or which predicted features to select
        # for the time being, lets take the max of the values predicted for each feature
        # however this seems really wrong.
        return logits

    def predict_from_single_context_vector(self, word, vec):
        # code from form_input bc we already have embedding
        x =  torch.from_numpy(vec).float()
        logits = self.nn.forward(x)
        logits = logits.detach().numpy()
        return (word, logits)

    def predict_top_n_features(self, word: str, n: int, vec=None):

        if vec is not None:
            logits = vec
        else:
            logits = self.predict(word)
    
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        # Newer NumPy versions (1.8 and up) have a function called argpartition for this. To get the indices of the four largest elements, do
        ind = np.argpartition(logits, -n)[-n:]

        feats = []
        for i in ind:
            feat = self.feature_norms.feature_map.get_object(i)
            feats.append(feat)
        return feats

    def predict_top_n_features_from_single_context_vector(self, word: str, n: int, input_vec, output_vec=None):
        """
        use this when we already have one context vector and we want to get semantic features for it. 
        for use with multi-prototype centroids, when the target vector is an aggregate of several context vectors already
        """
        if output_vec is not None:
            logits = output_vec
        else:
            word, logits = self.predict_from_single_context_vector(word, input_vec)
    
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        # Newer NumPy versions (1.8 and up) have a function called argpartition for this. To get the indices of the four largest elements, do
        ind = np.argpartition(logits, -n)[-n:]

        feats = []
        for i in ind:
            feat = self.feature_norms.feature_map.get_object(i)
            feats.append(feat)

        #print(feats)
        return (word, feats)

    def predict_in_context(self, word, sentence, bert, glove=False):
        if glove:
            return self.predict(word)

        # generate bert vector for word
        vec = bert.get_bert_vectors_for(word, sentence)
        # get the layer we care about
        vec = vec[8]

        # put it ias the only prototype in a bag
        vec = np.array([vec])

        # form input in context
        x =  torch.from_numpy(vec).float()
        logits = self.nn.forward(x)
        logits = logits.detach().numpy()

        return logits

    def predict_top_n_features_in_context(self, word, sentence, n, bert=None, vec=None, glove=False):
        """
        :vec: optional arg if we already have the predicted feature vector
        """

        if vec is not None:
            logits = vec
        else:
            logits = self.predict_in_context(word, sentence, bert, glove=glove)
    
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        # Newer NumPy versions (1.8 and up) have a function called argpartition for this. To get the indices of the four largest elements, do
        ind = np.argpartition(logits, -n)[-n:]

        feats = []
        for i in ind:
            feat = self.feature_norms.feature_map.get_object(i)
            feats.append(feat)
        return feats

class FrequencyClassifier(FeatureClassifier):
    """
    Person classifier that takes counts of how often a word was observed to be the positive and negative class
    in training, and classifies as positive any tokens which are observed to be positive more than negative.
    Unknown tokens or ties default to negative.
    Attributes:
        pos_counts: how often each token occurred with the label 1 in training
        neg_counts: how often each token occurred with the label 0 in training
    """
    def __init__(self, feature_norms):
        vecs = []
        self.feature_norms = feature_norms
        words = self.feature_norms.feature_norms.keys()
        for word in words:
            vec = feature_norms.get_feature_vector(word)
            vecs.append(vec)
            
        avg_features = np.average(vecs, axis=0)
        print(avg_features)

        ind = np.argpartition(avg_features, -10)[-10:]
        feats = []
        for i in ind:
            feat = feature_norms.feature_map.get_object(i)
            feats.append(feat)
        print(feats)
        
        self.prediction = avg_features
        

    def predict(self, word):
        return self.prediction


class BinaryClassifier(object):

    def __init__(self, nn, vectors, feature_norms):
        self.nn = nn
        self.word_vectors = vectors
        self.feature_norms = feature_norms


    def predict(self, word: str):
        """
        Makes feature predictions for the word  given by word
        :param word:
        :return: logits over all output classes (output sized vector)
        """

        x = form_input(word, self.word_vectors)
        logits = self.nn.forward(x)
        logits = logits.detach().numpy()

        logits = [1 if sigmoid(logit) > 0.5 else 0 for logit in logits]

        return logits

    def predict_top_n_features(self, word: str, n: int):
        logits = self.predict(word)
            
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        # Newer NumPy versions (1.8 and up) have a function called argpartition for this. To get the indices of the four largest elements, do
        ind = np.argpartition(logits, -n)[-n:]

        feats = []
        for i in ind:
            feat = self.feature_norms.feature_map.get_object(i)
            feats.append(feat)

        return feats


# Luong attention layer
class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        #energy = self.attn(encoder_output)
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class AttentionSoftMax(torch.nn.Module):
    def __init__(self, in_features = 1, out_features = None):
        """
        given a tensor `x` with dimensions [N * M],
        where M -- dimensionality of the feature vector
                   (number of features per instance)
              N -- number of instances
        initialize with `AggModule(M)`
        returns:
        - weighted result: [M]
        - gate: [N]
        """
        super(AttentionSoftMax, self).__init__()
        self.otherdim = ''
        if out_features is None:
            out_features = in_features
        self.layer_linear_tr = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU()
        self.layer_linear_query = nn.Linear(out_features, 1)
        
    def forward(self, x):
        keys = self.layer_linear_tr(x)
        keys = self.activation(keys)
        attention_map_raw = self.layer_linear_query(keys)[...,0]
        attention_map = nn.Softmax(dim=-1)(attention_map_raw)
        result = torch.einsum(f'{self.otherdim}i,{self.otherdim}ij->{self.otherdim}j', attention_map, x)
        return result, attention_map

class FFNN(nn.Module):
    """
    Defines the core neural network for doing multiclass classification over a single datapoint at a time. This consists
    of matrix multiplication, tanh nonlinearity, another matrix multiplication, and then
    a log softmax layer to give the ouputs. Log softmax is numerically more stable. If you take a softmax over
    [-100, 100], you will end up with [0, 1], which if you then take the log of (to compute log likelihood) will
    break.

    The forward() function does the important computation. The backward() method is inherited from nn.Module and
    handles backpropagation.
    """
    def __init__(self, inp, hid, out, bag_size, dropout):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        print("Initializing FFNN with input size %s, hidden size %s, output size %s" % (inp, hid, out))
        self.num_classes = out

        super(FFNN, self).__init__()
        self.V = nn.Linear(inp, hid)
        self.g = nn.Tanh()
        self.W = nn.Linear(hid, out)
        # Initialize weights according to a formula due to Xavier Glorot.
        self.dropout = nn.Dropout(p=dropout)

        # Initialize with zeros instead
        nn.init.zeros_(self.V.weight)
        nn.init.zeros_(self.W.weight)
        
        self.attn = AttentionSoftMax(inp , out_features = None)


        self.layers = nn.Sequential(
          nn.Linear(inp, hid),
          nn.ReLU(),
          nn.Linear(hid, hid),
          nn.ReLU(),
          nn.Linear(hid, hid),
          nn.ReLU(),
          nn.Dropout(p=dropout),
          nn.Linear(hid, out)
        )


    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        weighted_avg, A = self.attn(x)  # NxK
        return self.dropout(self.W(self.g(self.V(weighted_avg))))

def dot_score(self, hidden, encoder_output):
    return torch.sum(hidden * encoder_output, dim=2)


def form_input(word: str, embs: MultiProtoTypeEmbeddings):
    """
    returns the numpy BERT vector for that word
    """
    vec = embs.get_embedding(word)
    return torch.from_numpy(vec).float()


def form_output(word: str, norms: FeatureNorms, binary=False):
    """
    returns a NON-SPARSE numpy vector representing the buchanan feature norms for this word.
    """
    norm = norms.get_feature_vector(word)
    if binary == True:
        norm = [1 if val > 0 else 0 for val in norm]
    norm = torch.FloatTensor(norm)

    return norm

def train_ffnn(train_exs: List[str], dev_exs: List[str], multipro_embs: MultiProtoTypeEmbeddings, feature_norms: FeatureNorms, args) -> FeatureClassifier:
    num_epochs = args['epochs']
    initial_learning_rate = args['lr']
    hidden_size = args['hidden_size']
    multipro_vec_size = multipro_embs.dim
    num_classes = feature_norms.length
    num_bags = multipro_embs.num_prototypes
    dropout = args['dropout']

    ffnn = FFNN(multipro_vec_size, hidden_size, num_classes, num_bags, dropout)
    optimizer = optim.Adam(ffnn.parameters(), lr=initial_learning_rate)
    
    mse = nn.MSELoss(reduction='none')

    for epoch in range(0, num_epochs):
        
        ex_indices = [i for i in range(0, len(train_exs))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            x = form_input(train_exs[idx], multipro_embs)
            y = form_output(train_exs[idx], feature_norms)
            # Build one-hot representation of y. Instead of the label 0 or 1, y_onehot is either [0, 1] or [1, 0]. This
            # way we can take the dot product directly with a probability vector to get class probabilities.
            #y_onehot = torch.zeros(self.ffnn.num_classes)
            # scatter will write the value of 1 into the position of y_onehot given by y
            #y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            ffnn.zero_grad()


            log_probs = ffnn.forward(x)
            loss = mse(log_probs.unsqueeze(0), y.unsqueeze(0)).sum()
            total_loss += loss

            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()
        print("\nTotal loss on epoch %s: %f" % (epoch, total_loss))
        
        model = FeatureClassifier(ffnn, multipro_embs, feature_norms)
        model.nn.eval()
        print("=======TRAIN SET=======")
        evaluate(model, train_exs, feature_norms, args, debug='false')
        print("=======DEV SET=======")
        evaluate(model, dev_exs, feature_norms, args, debug='false')


    return model

def evaluate(model, dev_exs, feature_norms, args, debug='false'):
    y_hat = []
    y = []
    cosines = []
    top_10_precs = []
    top_20_precs = []
    top_k_precs = []
    correlations = []
    mses = []
    num_total = 0

    # we're calling this in the particular model trainer now bc this is now a more general function for more than ffnns
    #model.nn.eval()

    for i in range(0,len(dev_exs)):
        num_total +=1
        word = dev_exs[i]

        prediction = model.predict(word)
        y_hat.append(prediction)

        gold = feature_norms.get_feature_vector(word)
        gold_feats = feature_norms.get_features(word)
        y.append(gold)
        cos = 1 - cosine(prediction, gold)
        cosines.append(cos)


        top_10 = model.predict_top_n_features(word, 10, vec=prediction)
        top_10_gold = feature_norms.top_n(word, 10)

        num_in_top_10 = len(set(top_10).intersection(set(top_10_gold)))
        prec = num_in_top_10 / len(top_10_gold)
        top_10_precs.append(prec)

        top_20 = model.predict_top_n_features(word, 20, vec=prediction)
        top_20_gold = feature_norms.top_n(word, 20)

        num_in_top_20 = len(set(top_20).intersection(set(top_20_gold)))
        prec = num_in_top_20 / len(top_20_gold)
        top_20_precs.append(prec)

        gold_len = len(gold_feats)
        top_k = model.predict_top_n_features(word, gold_len, vec=prediction)
        num_in_top_k = len(set(top_k).intersection(set(gold_feats)))
        top_k_prec = num_in_top_k / gold_len
        top_k_precs.append(top_k_prec)

        corr, p = spearmanr(prediction, gold)
        correlations.append(corr)

        rsq = r2_score(gold, prediction)
        mse = mean_squared_error(gold,prediction)
        mses.append(mse)


        if ((i % 20 ==0) and debug=='info') or (debug=='true'):
            print(word)
            print("top ten predicted features: ", top_10)
            print("top ten gold features: ", top_10_gold)
            print("gold features: ", gold_feats)
            print("top k predicted features: ", top_k)

            print("cosine: %f" % cos)
            print("precison: %f" % prec)
            print("correlation: %f" % corr)
            print("top k acc: %f" % top_k_prec)
            print("coefficient of determination: %f" % rsq )


    top_10_prec = np.average(top_10_precs)
    top_20_prec = np.average(top_20_precs)
    top_k_prec = np.average(top_k_precs)
    average_correlation = np.average(correlations)
    average_cosine = np.average(cosines)
    mse = np.average(mses)

    R_square = r2_score(y, y_hat)

    print("Average cosine between gold and predicted feature norms: %s" % average_cosine)
    print("average Percentage (%) of gold gold-standard features retrieved in the top 10 features of the predicted vector: ", top_10_prec)
    print("average Percentage (%) of gold gold-standard features retrieved in the top 20 features of the predicted vector: ", top_20_prec)
    print("Average % @k (derby metric)", top_k_prec)
    print("correlation between gold and predicted vectors: %s " % average_correlation)
    print('Coefficient of Determination', R_square)
    print("MSE: %f" % mse )


    results =  {
        "MAP_at_10": top_10_prec, 
        "MAP_at_20": top_20_prec, 
        "MAP_at_k": top_k_prec, 
        "correl": average_correlation, 
        "cos": average_cosine, 
        "rsquare": R_square, 
        "mse": mse }
    return results