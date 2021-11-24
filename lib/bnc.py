import os.path
import csv
import nltk.corpus.reader.bnc


class BNC():


    def __init__(self):
        self.corpus = nltk.corpus.reader.bnc.BNCCorpusReader(root='./data/BNC/Texts/', fileids=r'[A-K]/\w*/\w*\.xml')
        self.length = bnc_length()


class BNCToken():
    def __init__(self, sentence, pos, uid, vector=None):
        self.sentence = sentence
        self.pos = pos
        self.uid = uid
        self.vector = vector

class BNCWord():

    def __init__(self, word: str, tokens: [BNCToken]):
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

def randomly(seq, pseudo=True):
    import random
    shuffled = list(seq)  
    if pseudo:
        seed = lambda : 0.479032895084095295148903189394529083928435389203890819038471
        random.shuffle(shuffled, seed)
    else:
        print("shuffling indexes")
        random.shuffle(shuffled) 
        print("done shuffling")
    return list(shuffled)


def collect_bnc_tokens_for_words(words, max_num_examples, outfile, override=False):

    
    #filename = outfile
    #pathname = os.path.join(parent_dir, filename)  
    
    # do we already have the data collected?
    if os.path.isfile(outfile) and override==False:
        print("data already exist at %s" % outfile)
        return
    
    else:    
        bnc = BNC()
        corpus = bnc.corpus.tagged_sents(strip_space=True)
        corpus_length = bnc.length
        print("# Sentences in BNC corpus: %s" % corpus_length)

        
        with open(outfile, mode='w') as out:
            writer = csv.writer(out, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
            
            # create a data structure for keeping tabs on how many tokens we have collected
            unigrams = {}
            for i in range(0, len(words)):
                    word = words.get_object(i)
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

def sort_bnc_tokens(big_long_file, embeddings_dir, words) -> [BNCWord]:
    """
    # you already have tokens collected for each word 
    # now these tokens ought to be sorted into their own files

    # ensure that there is a word_data directory to store in our words
    # you have to delete it first with rm -rf if we are reloading
    """
    os.mkdir(embeddings_dir)


    # create files for each word we care about
    for i in range(0, len(words)):
        word = words.get_object(i)
        word_dir = os.path.join(embeddings_dir, word)
        os.mkdir(word_dir)


    # read in the big long file
    with open(big_long_file, mode="r") as infile:
        fieldnames = ["word", "sentence", "POS", "id"]
        reader = csv.DictReader(infile, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC, fieldnames=fieldnames)
        
        # split the big long file into smaller, sorted files that are easier to process one at a time
        for row in reader:
            
            word = row["word"]
            text = row["sentence"]
            pos = row["POS"]
            uid = "BNC_" + str(int(row["id"]))

            # open file for this word to spit tokens into
            token_file = os.path.join(embeddings_dir, word, "BNC_tokens.csv")
            with open(token_file, mode="a") as outfile:
                # finally, write all of the info with the vector to disk
                writer = writer = csv.writer(outfile, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow([word, text, pos, uid])

def read_tokens_for(word, data_dir='data/multipro_embeddings'):
    """
    # read in the tokens for this word
    """
    tokens = []
    try:
        pathname = os.path.join(data_dir, word, 'BNC_tokens.csv')
        with open(pathname, mode='r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter='\t', fieldnames=["word", "sentence", "tag", "uid"])  
            tokens = [BNCToken(row['sentence'], row['tag'], row['uid']) for row in reader]
            data = BNCWord(word, tokens)
            return data
    except: None