# models.py

import time
import numpy as np
from utils import *
from collections import Counter
from multiprototype import *
from feature_data import *
import torch.nn as nn
from torch import optim
import random



from typing import List

def add_models_args(parser):
    """
    Command-line arguments to the system related to your model.  Feel free to extend here.  
    """
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=10, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--hidden_size', type=int, default=300, help='size of hidden state')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes
    #parser.add_argument('--embedding_size', type=int, default=50, help='size of embedding to train')
    #parser.add_argument('--pretrained', type=bool, default=True, help='Boolean indicating whether to start with pretrained vectors')
    #parser.add_argument('--embedding_dropout', type=float, default=0.2, help='Embedding dropout probability')


# class DumbClassifier(object):
#     """
#     Person classifier that takes counts of how often a word was observed to be the positive and negative class
#     in training, and classifies as positive any tokens which are observed to be positive more than negative.
#     Unknown tokens or ties default to negative.
#     Attributes:
#         pos_counts: how often each token occurred with the label 1 in training
#         neg_counts: how often each token occurred with the label 0 in training
#     """
#     def __init__(self, pos_counts: Counter, neg_counts: Counter):
#         self.pos_counts = pos_counts
#         self.neg_counts = neg_counts

#     def predict(self, tokens: List[str], idx: int):
#         if self.pos_counts[tokens[idx]] > self.neg_counts[tokens[idx]]:
#             return 1
#         else:
#             return 0


# def train_count_based_binary_classifier(ner_exs: List[PersonExample]) -> CountBasedPersonClassifier:
#     """
#     :param ner_exs: training examples to build the count-based classifier from
#     :return: A CountBasedPersonClassifier using counts collected from the given examples
#     """
#     pos_counts = Counter()
#     neg_counts = Counter()
#     for ex in ner_exs:
#         for idx in range(0, len(ex)):
#             if ex.labels[idx] == 1:
#                 pos_counts[ex.tokens[idx]] += 1.0
#             else:
#                 neg_counts[ex.tokens[idx]] += 1.0
#     print("All counts: " + repr(pos_counts))
#     print("Count of Peter: " + repr(pos_counts["Peter"]))
#     return CountBasedPersonClassifier(pos_counts, neg_counts)


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
        return logits

    def predict_top_ten_features(self, word: str):
        logits = self.predict(word)
        logits = logits.detach().numpy()
    
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        # Newer NumPy versions (1.8 and up) have a function called argpartition for this. To get the indices of the four largest elements, do
        ind = np.argpartition(logits, -10)[-10:]

        feats = []
        for i in ind:
            feat = self.feature_norms.feature_map.get_object(i)
            feats.append(feat)

        #print(feats)
        return feats

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
    def __init__(self, inp, hid, out):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        print("Initializing FFNN with input size %s, hidden size %s, output size %s" % (inp, hid, out))
        super(FFNN, self).__init__()
        self.V = nn.Linear(inp, hid)
        self.g = nn.Tanh()
        #self.g = nn.ReLU()
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)
        self.num_classes = out
        # Initialize with zeros instead
        # nn.init.zeros_(self.V.weight)
        # nn.init.zeros_(self.W.weight)


    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        #return self.log_softmax(self.W(self.g(self.V(x))))
        return self.W(self.g(self.V(x)))


def form_input(word: str, embs: MultiProtoTypeEmbeddings):
    """
    returns the numpy BERT vector for that word
    """
    vec = embs.get_embedding(word)

    """
    TODO implement bag. for now just average things together
    """
    vec = np.average(vec, axis=0)


    vec =  torch.from_numpy(vec).float()

    return vec


def form_output(word: str, norms: FeatureNorms):
    """
    returns a NON-SPARSE numpy vector representing the buchanan feature norms for this word.
    """
    #norm = norms.print_features(word)
    norm = norms.get_feature_vector(word)
    norm = torch.from_numpy(norm).float()
    #norm = norm.unsqueeze(0) # add dummy batch dimension
    #print(norm)
    return norm

def train_classifier(train_exs: List[str], dev_exs: List[str], multipro_embs: MultiProtoTypeEmbeddings, feature_norms: FeatureNorms, args) -> FeatureClassifier:
    num_epochs = args.epochs
    batch_size = args.batch_size
    initial_learning_rate = args.lr
    hidden_size = args.hidden_size
    multipro_vec_size = multipro_embs.dim
    num_classes = feature_norms.length

    ffnn = FFNN(multipro_vec_size, hidden_size, num_classes)

    #train_xs = [sentence_vector(ex.words, self.word_vectors) for ex in train_exs]
    #train_ys = [ex.label for ex in train_exs]

    optimizer = optim.Adam(ffnn.parameters(), lr=initial_learning_rate)
    multi_criterion = nn.MultiLabelSoftMarginLoss(weight=None, reduction='none')

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

            # Can also use built-in NLLLoss as a shortcut here but we're being explicit here
            #loss = torch.neg(log_probs).dot(y)
            loss = multi_criterion(log_probs.unsqueeze(0), y.unsqueeze(0)) # add dummy batch dimension
            total_loss += loss

            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()
        print("Total loss on epoch: %f" % (total_loss))
        
        model = FeatureClassifier(ffnn, multipro_embs, feature_norms)
        evaluate(model, dev_exs, feature_norms, args)


    return model

def evaluate(model, dev_exs, feature_norms, args):
    outputs = []
    for i in range(0,len(dev_exs)):
        word = dev_exs[i]
        top_10 = model.predict_top_ten_features(word)

        top_10_gold = feature_norms.top_10(word)

        if (i % 30 ==0) and args.print_dataset:
            print(word)
            print(top_10)
            print(top_10_gold)

    #raise Exception("what are we doingggg")

def print_evaluation(golds: List[int], predictions: List[int]):
    """
    Prints evaluation statistics comparing golds and predictions, each of which is a sequence of 0/1 labels.
    Prints accuracy as well as precision/recall/F1 of the positive class, which can sometimes be informative if either
    the golds or predictions are highly biased.

    :param golds: gold labels
    :param predictions: pred labels
    :return:
    """
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1
    acc = float(num_correct) / num_total
    output_str = "Accuracy: %i / %i = %f" % (num_correct, num_total, acc)
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    output_str += ";\nPrecision (fraction of predicted positives that are correct): %i / %i = %f" % (num_pos_correct, num_pred, prec)
    output_str += ";\nRecall (fraction of true positives predicted correctly): %i / %i = %f" % (num_pos_correct, num_gold, rec)
    output_str += ";\nF1 (harmonic mean of precision and recall): %f;\n" % f1
    print(output_str)
    return acc, f1, output_str

