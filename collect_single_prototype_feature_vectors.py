from bert import *
from feature_data import *
from bnc import *
from multiprototype import *


# get words to collect single prototype vectors for
exs = read_cue_feature_examples('data/buchanan/cue_feature_words.csv')
unique_words = unique_words(exs)
unique_words = [unique_words.get_object(i) for i in range(0, len(unique_words))]

# debug: just do 5
# unique_words = [unique_words.get_object(i) for i in range(5)]


# collect tokens for each word

outfile = 'data/bnc_words_with_context_tokens.csv'
#max_num_examples = 200
#collect_bnc_tokens_for_words(unique_words, max_num_examples, outfile)

embeddings_dir = 'data/multipro_embeddings'
#sort_bnc_tokens(outfile, embeddings_dir, unique_words)

#initialize BERT model
#bert_base = BERTBase()


layers = [8]
cluster_sizes = [1,5]

# Generate clusters
# NOTE: already done for layer 8, clusters 1,5
# DEBUG: unique_words = [unique_words.get_object(i) for i in range(1)]
# generate_clusters(bert_base, embeddings_dir, unique_words, layers, cluster_sizes)

#Compile clusters together into text embedding files
for layer in layers:
    for cluster_size in cluster_sizes:
        output_dir = './data/multipro_embeddings/layer' + str(layer) + 'clusters' + str(cluster_size) + '.txt'
        print("compiling multiprototype embedding textfiles for layer %s with %s clusters" % (layer, cluster_size))
        embs = prepare_embeddings(embeddings_dir, output_dir, unique_words, layer, cluster_size)

# read in the layer 8 5-prototype embeddings
embs = read_multiprototype_embeddings('./data/multipro_embeddings/layer8clusters5.txt', layer=8, num_clusters=5)
#print(embs)

leaving = embs.get_embedding('leaving')
print(leaving)


# DEBUG try the first few
norms = FeatureNorms(exs)
#print(norms.feature_norms)
print(norms.length)
#read_feature_norms('data/buchanan/cue_feature_words.csv')