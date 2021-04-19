import os.path
import csv

class BNC():


    def __init__(self):
        corpus = bnc.BNCCorpusReader(root='../data/BNC/Texts/', fileids=r'[A-K]/\w*/\w*\.xml')
        length = bnc_length()

class BNCWord():

    def __init__(self, word: str, tokens: [str]):
        """
        wrapper around a word and a list of sentences containing that word
        """
        self.word = word
        self.tokens = tokens


def bnc_length(pathname='data/bnc_length.txt'):
    try:
        with open(pathname, 'r') as fh:
            count = int(fh.read())
            return count
    except:
        print("BNC not yet indexed. Calculating length and writing to 'data/count_of_bnc_sentences.txt'")
        bnc_reader = get_bnc()
        corpus = bnc_reader.tagged_sents(strip_space=True)
        length = len(corpus)
        with open(pathname, 'w') as disk:
            disk.write(str(length))
        return length

def bnc_sentence_to_string(sentence):
    words = [word.lower() for (word, pos) in sentence]
    return " ".join(words)








def collect_bnc_tokens_for_words(words, max_num_examples, override=False, outfile):

    
    #filename = outfile
    #pathname = os.path.join(parent_dir, filename)  
    
    # do we already have the data collected?
    if os.path.isfile(outfile) and override==False:
        print("data already exist at %s" % pathname)
        return
    
    else:    
        bnc_reader = datasets.get_bnc()
        corpus = bnc_reader.tagged_sents(strip_space=True)
        corpus_length = datasets.bnc_length()
        print("# Sentences in BNC corpus: %s" % corpus_length)

        
        with open(pathname, mode='w') as outfile:
            writer = csv.writer(outfile, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
            
            # create a data structure for keeping tabs on how many tokens we have collected
            unigrams = {}
            for word in words:
                    unigrams[word]=max_num_examples                    
            
            # come up with a random order in which to traverse the BNC
            randomized_indexes = randomly([x for x in range(corpus_length)], pseudo=False)
            print(randomized_indexes[:50])
            
            
            """"
            Iterate through the corpus, looking at words one by one, and 
            keep iterating as long as we still have tokens to collect
            """
            i = 0
            
            while (unigrams and randomized_indexes):
                # track progress
                i+=1
                if i % 100000 == 0:
                    print("Processed %s sentences" % i)
                
                # fetch the next random sentence
                corpus_index = randomized_indexes.pop()
                sentence = corpus[corpus_index]
               
            
                # keep track of words we've seen in this sentence, so we don't collect
                # a word twice if it appears twice in the sentence. 
                seen_words = set()
                
                for word_tuple in sentence:
                    word = word_tuple[0].lower()
                    tag = word_tuple[1]

                    token_count = unigrams.get(word) 
                    
                    # collect this sentence as a token of the word
                    if (token_count != None) and (word not in seen_words):

                        string = ' '.join([w[0] for w in sentence])
                        
                        if i % 100000 == 0:
                            print(word)
                            print(tag)
                            print(string)
                            print(corpus_index)
                        
                        writer.writerow([word, string, tag, corpus_index])
                        seen_words.add(word)
                        if unigrams[word]==0:
                            del unigrams[word]
                        else:
                            unigrams[word] -=1

# read in the tokens for this word
def read_tokens_for(word, data_dir='../data/word_data'):
    try:
        pathname = os.path.join(data_dir, word, 'BNC_tokens.csv')
        with open(pathname, mode='r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter='\t', fieldnames=["word", "sentence", "tag", "uid"])  
            data = [row for row in reader]
            return data
    except: None