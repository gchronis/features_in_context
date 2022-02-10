from typing import List
from multiprototype import *
from feature_data import *
from models import FeatureClassifier, BinaryClassifier, form_input, evaluate
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial.distance import cosine




class KNNRegressor(FeatureClassifier):
#     """
#     Classifier to classify predict a distribution over features for a bert word-type embedding
#     """

    def __init__(self, trained_models, vectors, feature_norms):
        """
        Initializes a label spreading model

        :param models:          list of trained label prop models, one for each semantic feature in the feature norms
                                same length as feature norms
        :param vectors:         the input embeddings
        :param feature_norms:   the gold embeddings; instance of FeatureNorms
        """
        self.models = trained_models
        self.num_features = len(trained_models)
        self.word_vectors = vectors
        self.feature_norms = feature_norms
        self.num_models = feature_norms.length




    def predict(self, word: str):
        """
        Makes feature predictions for the word  given by word
        :param word: the word we want to predict the features for

        :return: logits over all output classes (output sized vector)
        """
        logits = []

        for i in range (0, self.num_features):

            # this gives a bag of prototypes of shape (5, 3981)
            x = form_input(word, self.word_vectors)

            # since we dont have an MIL setup for PLSR, we just average the prototype predictions. We can also average after the prediction, this can be another option
            x = np.average(x, axis=0)

            # the model needs a vertical vector
            x = x.reshape(1, -1)

            #print("predicting features for ", word)
            #print("vector with shape ", x.shape)
            pred = self.models[i].predict(x)
            #print("prediction is ", pred[0])

            #print(logits)
            #print(pred)
            logits.append(pred[0])
            #print(logits)
        

        #print("predicted_vector_for ", word)
        #print(logits)
        return logits

    def predict_top_n_features(self, word: str, n: int, vec=None):
 
        if vec:
            logits = vec
        else:
            logits = self.predict(word)
        #logits = logits.detach().numpy()
    
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        # Newer NumPy versions (1.8 and up) have a function called argpartition for this. To get the indices of the four largest elements, do
        ind = np.argpartition(logits, -n)[-n:]

        feats = []
        for i in ind:
            feat = self.feature_norms.feature_map.get_object(i)
            feats.append(feat)

        #print(feats)
        return feats


    def predict_in_context(self, word, sentence, bert):
        # generate bert vector for word
        vec = bert.get_bert_vectors_for(word, sentence)
        # get the layer we care about
        vec = vec[8]
        #print(vec.shape)

        logits = []
        for i in range (0, self.num_features):
            pred = self.models[i].predict(vec)
        
        logits = logits.append(pred)

        return logits

    def predict_top_n_features_in_context(self, word, sentence, n, bert=None, vec=None):

        if vec:
            logits = vec
        else:
            logits = self.predict_in_context(word, sentence, bert)
        #logits = logits.detach().numpy()
    
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        # Newer NumPy versions (1.8 and up) have a function called argpartition for this. To get the indices of the four largest elements, do
        ind = np.argpartition(logits, -n)[-n:]

        feats = []
        for i in ind:
            feat = self.feature_norms.feature_map.get_object(i)
            feats.append(feat)

        #print(feats)
        return feats   

        #print(feats)


        res.predict()

    # predict
    # predict in context
    # predict top n features
    # predict topn features in context.   

        #print(feats)


    def evaluate_binary(self, dev_exs, debug='false'):

        y_hat = []
        y = []
        cosines = []
        precs = []
        correlations = []

        for i in range(0,len(dev_exs)):
            word = dev_exs[i]

            prediction = self.predict(word)


            
            y_hat.append(prediction)


            gold = self.feature_norms.get_feature_vector(word)
            y.append(gold)

            print(prediction)
            print(gold)

            cos = cosine(prediction, gold)
            cosines.append(cos)

            corr, p = spearmanr(prediction, gold)
            correlations.append(corr)

            if (i % 30 ==0) and debug=='info':
                print(word)
                print(prediction)
                print(gold)
                print("cosine: %f" % cos)
                #print("precison: %f" % prec)
                print("correlation: %f" % corr)


def column(matrix, i):
    return [row[i] for row in matrix]

def train_knn_regressor(train_exs: List[str], dev_exs: List[str], multipro_embs: MultiProtoTypeEmbeddings, feature_norms: FeatureNorms, args) -> KNNRegressor:
    num_epochs = args.epochs
    batch_size = args.batch_size
    initial_learning_rate = args.lr
    hidden_size = args.hidden_size
    multipro_vec_size = multipro_embs.dim
    num_classes = feature_norms.length

    """
    First prepare a dataset with our labeled examples
    (all of the instances , labeled with the features)
    """
    # for each of the train examples, make an instance of all the prototypes and label them with the gold vector
    X = []
    y = []


    # iterate through the words

    ### DEBUG only do a few train exs
    for word in train_exs:
        # look up the embedding
        emb = multipro_embs.get_embedding(word)
        # look up the feature norm
        norm = feature_norms.get_feature_vector(word)
        # convert to binary feature vector
        norm = [1 if val > 0 else 0 for val in norm]

        # iterate through the prototypes for each word
        #print(emb.shape)        
        for index in range(0, emb.shape[0]):

            # add vector to training xs
            vector = emb[index, :]
            #print(vector.shape)
            X.append(vector)


            # add feature norm to training ys
            #print(norm.shape)
            #print("first feature: ", norm[0])
            y.append(norm)

    print("training length")
    print(len(X))
    print(len(y))


    """
    we need to create and train a model for each feature in the embeddings.
    For each model, we have to construct a new gold dataset from the entire gold feature set, which has
    just feature x for all the words. Just the "red" feature, etc.
    """
    trained_models = []

    # iterate through the semantic features and fit a model for each one
    for i in range(0, feature_norms.length):
    
    # DEBUG: just do the first 10 features
    #for i in range(0, 10):
        feature = feature_norms.feature_map.get_object(i)
        
        if i % 30 == 0:
            print("feature", i)
            print("running knn for feature: ", feature)

        this_feature_for_all_words = column(y, i)

        #print(len(y))
        #print(y)

        #print(this_feature_for_all_words)


        """
        TODO add in the options here from args rather than using the defaults
        """

        # we want a cosine distance metric, so we have to use out own.
        # combining the distance metric and the weights into one function call. but we could put these into separate parameters in knn regressor
        # initialization as 'metric' and 'weights'
        """
        the power of is the lambda parameter from Johns and Jones
        """
        cosine_metric = lambda x,y: pow(cosine(x,y), 10)

        model = KNeighborsRegressor(n_neighbors=20, n_jobs=-1, metric=cosine_metric)
        #print(len(X))
        #print(len(this_feature_for_all_words))

        res = model.fit(X, this_feature_for_all_words)
        trained_models.append(res)
    
    """
    we also need i think our unlabeled examples?
    """



    model  = KNNRegressor(trained_models, multipro_embs, feature_norms)



    #print("=======EVAL ON TRAIN SET=======")
    #evaluate(model, train_exs, feature_norms, args, debug='false')
    print("=======EVAL ON DEV SET=======")
    evaluate(model, dev_exs, feature_norms, args, debug='info')

    return model

    # okqy so how does this woek, you need to make a classifier that has which methods.

    # predict
    # predict in context
    # predict top n features
    # predict topn features in context. 
