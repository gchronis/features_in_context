from typing import List
from src.multiprototype import *
from src.feature_data import *
from src.models import FeatureClassifier, evaluate, form_input
from sklearn.cross_decomposition import PLSRegression





class PLSRClassifier(FeatureClassifier):
#     """
#     Classifier to classify predict a distribution over features for a bert word-type embedding
#     """

    def __init__(self, model, vectors, feature_norms):
        """
        Initializes a label spreading model

        :param models:          list of trained label prop models, one for each semantic feature in the feature norms
                                same length as feature norms
        :param vectors:         the input embeddings
        :param feature_norms:   the gold embeddings; instance of FeatureNorms
        """
        self.model = model
        self.word_vectors = vectors
        self.feature_norms = feature_norms
        self.num_models = feature_norms.length


    def predict(self, word: str):
        """
        Makes feature predictions for the word  given by word
        :param word: the word we want to predict the features for

        :return: logits over all output classes (output sized vector)
        """

        # this gives a bag of prototypes of shape (5, 3981)
        x = form_input(word, self.word_vectors)

        # since we dont have an MIL setup for PLSR, we just average the prototype predictions. We can also average after the prediction, this can be another option
        x = np.average(x, axis=0)

        # the model needs a vertical vector
        x = x.reshape(1, -1)

        #print("predicting features for ", word)
        #print("vector with shape ", x.shape)
        logits = self.model.predict(x)

        # and it returns a verticla vector. our code wants it to be horizontal again, so we reshape again
        logits = logits[0]
        #print("predicted_vector_for ", word)
        #print(logits.shape)
        return logits


    def predict_in_context(self, word, sentence, bert, glove=False):
        if glove:
            return self.predict(word)
            
        # generate bert vector for word
        vec = bert.get_bert_vectors_for(word, sentence)
        # get the layer we care about
        vec = vec[8]

        # reshape to be vertical
        vec = vec.reshape(1, -1)
        #print("after reshape")
        #print(vec.shape)


        logits = self.model.predict(vec)
        #print("after prediction")
        #print(vec.shape)

        logits = logits[0]
        #print("after second reshape")
        #print(vec.shape)

        return logits

    # def predict_top_n_features_in_context(self, word, sentence, n, bert=None):
    #     logits = self.predict_in_context(word, sentence, bert)
    #     #logits = logits.detach().numpy()
    
    #     # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    #     # Newer NumPy versions (1.8 and up) have a function called argpartition for this. To get the indices of the four largest elements, do
    #     ind = np.argpartition(logits, -n)[-n:]

    #     feats = []
    #     for i in ind:
    #         feat = self.feature_norms.feature_map.get_object(i)
    #         feats.append(feat)

    #     #print(feats)
    #     return feats   

        #print(feats)



def train_plsr(train_exs: List[str], dev_exs: List[str], multipro_embs: MultiProtoTypeEmbeddings, feature_norms: FeatureNorms, args) -> PLSRClassifier:
    #num_epochs = args.epochs
    #batch_size = args.batch_size
    #initial_learning_rate = args.lr
    #hidden_size = args.hidden_size
    #multipro_vec_size = multipro_embs.dim
    #num_classes = feature_norms.length

    """
    First prepare a dataset with our labeled examples
    (all of the instances , labeled with the features)
    """
    # for each of the train examples, make an instance of all the prototypes and label them with the gold vector
    X = []
    y = []


    # iterate through the words
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


    plsr = PLSRegression(n_components=args['plsr_n_components'], max_iter=args['plsr_max_iter'], scale=True)
    plsr.fit(X, y)

    model = PLSRClassifier(plsr, multipro_embs, feature_norms)


    print("=======EVAL ON TRAIN SET=======")
    evaluate(model, train_exs, feature_norms, args, debug='false')
    print("=======EVAL ON DEV SET=======")
    evaluate(model, dev_exs, feature_norms, args, debug='info')

    return model
    # okqy so how does this woek, you need to make a classifier that has which methods.

    # predict
    # predict in context
    # predict top n features
    # predict topn features in context. 
