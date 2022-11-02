#!pip install pytorch-pretrained-bert

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np

class BERTBase():

    def __init__(self):
        # Load pre-trained model (weights)
        model = BertModel.from_pretrained('bert-base-uncased')
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()
        self.model = model

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def get_bert_vectors_for(self, word, text):
        """
        Run the token sentence through the model and calculate a word vector
        based on the mean of the WordPiece vectors in the last layer
        """
        tokenized_word = self.tokenizer.tokenize(word)

        # Add the special tokens.
        marked_text = "[CLS] " + text + " [SEP]"
        # Split the sentence into tokens.
        tokenized_text = self.tokenizer.tokenize(marked_text)
        segments_ids = [1] * len(tokenized_text)

        # Map the token strings to their vocabulary indeces.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Display the words with their indeces.
        #for tup in zip(tokenized_text, indexed_tokens):
            #print('{:<12} {:>6,}'.format(tup[0], tup[1]))

        # Mark each of the tokens as belonging to sentence "1".
        segments_ids = [1] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        try:
            # Predict hidden states features for each layer
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
        except:
            print("tokenized sequence too long")
            print(tokenized_text)
            return None

        # Rearrange hidden layers to be grouped by token
        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings.size()

        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings.size()


        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,0,2)
        token_embeddings.size()
        
        vectors = []
        
        # get a vector for each layer of the network
        for layer in range(12):
            piece_vectors = []
            for word_piece in tokenized_word:
                # TODO should be the matching slice, because this doesnt account for repeat word  pieces
                index = tokenized_text.index(word_piece)
                token = token_embeddings[index]
                # `token` is a [12 x 768] tensor

                # Sum the vectors from the last four layers.
                #sum_vec = torch.sum(token[-4:], dim=0)

                # Use the vectors from the current layer
                vec = token[layer]
                piece_vectors.append(vec.numpy())

            # use the mean of all of the word_pieces. 
            layer_vector = np.average(piece_vectors, axis=0)    
        
            # add the vector for this layer to our grand list
            vectors.append(layer_vector)
        return vectors


if __name__=="__main__":

    import sys
    query_word_1 = sys.argv[1]
    query_sentence_1 = sys.argv[2]
    query_word_2 = sys.argv[3]
    query_sentence_2 = sys.argv[4]


    bert_base = BERTBase()

    emb1 = bert_base.get_bert_vectors_for(query_word_1, query_sentence_1)
    emb2 = bert_base.get_bert_vectors_for(query_word_2, query_sentence_2)

    emb1 = emb1[8]
    emb2 = emb2[8]
    # if embs.word_indexer.index_of(query_word_1) == -1:
    #     print("%s is not in the indexer" % query_word_1)
    # elif embs.word_indexer.index_of(query_word_2) == -1:
    #     print("%s is not in the indexer" % query_word_2)
    # else:
    #     emb1 = embs.get_embedding(query_word_1)
    #     emb2 = embs.get_embedding(query_word_2)
    print("cosine similarity of %s and %s: %f" % (query_word_1, query_word_2, np.dot(emb1, emb2)/np.sqrt(np.dot(emb1, emb1) * np.dot(emb2, emb2))))
