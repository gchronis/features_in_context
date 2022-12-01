import torch
from transformers import AutoTokenizer, AutoModel

class BERTBase():

    def __init__(self):
        model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def get_bert_vectors_for(self, word, text, word_occurrence=0):
        """
        Run the token sentence through the model and calculate a word vector
        based on the mean of the WordPiece vectors in the last layer
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        words = [i[0]
                for i in self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)]
        target_word_indices = [i for i, x in enumerate(words) if x == word]
        encoded_text = self.model(
            **inputs, output_hidden_states=True)["hidden_states"]
        word_start, word_end = inputs.word_to_tokens(target_word_indices[word_occurrence])
        avg_vectors_for_target_word = torch.cat(encoded_text)[:,word_start:word_end,:].mean(dim=1)
        return avg_vectors_for_target_word
