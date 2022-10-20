import argparse
import random
import numpy as np
from src.models import *
from src.label_propagation import *
from src.plsr import *
from src.knn import *
from src.modabs import *
from src.feature_data import *
from src.multiprototype import *
from src.utils import *
from typing import List
import time
from torch.utils.data import random_split
import torch
from ray import tune
import scipy.stats as stats




def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='main.py')
    
    # General system running and configuration options
    parser.add_argument('--do_dumb_thing', dest='do_dumb_thing', default=False, action='store_true', help='run the nearest neighbor model')
    #parser.add_argument('--train_path', type=str, default='data/geo_train.tsv', help='path to train data')
    #parser.add_argument('--dev_path', type=str, default='data/geo_dev.tsv', help='path to dev data')
    #parser.add_argument('--test_path', type=str, default='data/geo_test.tsv', help='path to blind test data')
    #parser.add_argument('--test_output_path', type=str, default='geo_test_output.tsv', help='path to write blind test results')
    #parser.add_argument('--domain', type=str, default='geo', help='domain (geo for geoquery)')
    parser.add_argument('--train_data', type=str, default='all', help='mc_rae_real mc_rae_subset buchanan binder (TODO mc_rae_animals) (TODO vinson_vigliocco)')
    parser.add_argument('--print_dataset', dest='print_dataset', default=False, action='store_true', help="Print some sample data on loading")
    parser.add_argument('--model', type=str, default='ffnn', help='ffnn (binary) frequency modabs label_propagation knn plsr')
    parser.add_argument('--layer', type=int, default=8, help='layer of BERT embeddings to use')
    parser.add_argument('--clusters', type=int, default=1, help='number of lexical prototypes in each BERT multi-prototype embedding')
    parser.add_argument('--save_path', type=str, default=None, help='path for saving model')
    parser.add_argument('--embedding_type', type=str, default='bert', help='glove word2vec bert')

    parser.add_argument('--k_fold', dest='k_fold', type=int, default=False, help='train using k-fold cross validation')
    parser.add_argument('--dev_equals_train', dest='dev_equals_train', default=False, action='store_true', help='use the training words as dev set for debug')
    parser.add_argument('--allbuthomonyms', dest='allbuthomonyms', default=False, action='store_true', help='train on all available words except for homonyms')
    #parser.add_argument('--allbutall', dest='allbutall', default=False, action='store_true', help='train on all available words')

    parser.add_argument('--tuning', default=False, action='store_true', help="writes stats to tuning file; does not save trained model")
    parser.add_argument('--zscore', default=False, action='store_true', help="zscore postprocessing on embeddings")


    add_models_args(parser) # defined in models.py

    args = parser.parse_args()


    return args


# def evaluate_classifier(exs: List[FeatureNorm], classifier, errors=False):
#     """
#     Prints evaluation of the classifier on the given examples
#     :param exs: PersonExample instances to run on
#     :param classifier: classifier to evaluate
#     """
#     predictions = []
#     golds = []
#     false_negs = {}
#     for ex in exs:
#         for idx in range(0, len(ex)):
#             gold = ex.labels[idx]
#             prediction = classifier.predict(ex, idx)
#             golds.append(gold)
#             predictions.append(prediction)

#             if prediction == 0 and gold == 1:
#                 false_negs[ex.tokens[idx]] = " ".join(ex.tokens)

#     if errors:
#         for key, value in false_negs.items():
#             print(key, ": ", value)

#     print_evaluation(golds, predictions)


# def print_evaluation(golds: List[int], predictions: List[int]):
#     """
#     Prints statistics about accuracy, precision, recall, and F1
#     :param golds: list of {0, 1}-valued ground-truth labels for each token in the test set
#     :param predictions: list of {0, 1}-valued predictions for each token
#     :return:
#     """
#     num_correct = 0
#     num_pos_correct = 0
#     num_pred = 0
#     num_gold = 0
#     num_total = 0
#     if len(golds) != len(predictions):
#         raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
#     for idx in range(0, len(golds)):
#         gold = golds[idx]
#         prediction = predictions[idx]
#         if prediction == gold:
#             num_correct += 1
#         if prediction == 1:
#             num_pred += 1
#         if gold == 1:
#             num_gold += 1
#         if prediction == 1 and gold == 1:
#             num_pos_correct += 1
#         num_total += 1
#     print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
#     prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
#     rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
#     f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
#     print("Precision: %i / %i = %f" % (num_pos_correct, num_pred, prec))
#     print("Recall: %i / %i = %f" % (num_pos_correct, num_gold, rec))
#     print("F1: %f" % f1)


def predict_write_output_to_file(exs: List[FeatureNorm], classifier, outfile: str):
    """
    Runs prediction on exs and writes the outputs to outfile, one token per line
    :param exs:
    :param classifier:
    :param outfile:
    :return:
    """
    f = open(outfile, 'w')
    for ex in exs:
        for idx in range(0, len(ex)):
            prediction = classifier.predict(ex, idx)
            f.write(ex.tokens[idx] + " " + repr(int(prediction)) + "\n")
        f.write("\n")
    f.close()

def prepare_data(feature_norms: FeatureNorms, embeddings: MultiProtoTypeEmbeddings, args):
    words = list(feature_norms.vocab.keys())
    #### DEBUG #####
    # toy training set
    # words = words[:100]
    ###############

    if args['allbuthomonyms']:
        ambiguous_pairs = feature_norms.ambiguous_pairs

        print("training on all words but evaluation homonyms")
        eval_words = [item for t in ambiguous_pairs for item in t]

        print("Starting with %s words" % len(words))
        train = [i for i in words if i not in eval_words]
        random.shuffle(train)
        val = eval_words
        test = eval_words

        print("Ending up with %s training words" % len(train))

    # elif args['allbutall']:
    #     train = words
    #     random.shuffle(train)
    #     val = train[:-10]
    #     test = train[:-10]


    else:

        validation_split = .1
        random_seed = 42

        dataset_size = len(words)
        split = int(np.floor(validation_split * dataset_size))
        #print(words)
        val, test, train = random_split(words, [split, split, dataset_size - split * 2 ], generator=torch.Generator().manual_seed(random_seed))
        print("Starting with %s training words" % len(train))

        #print(len(data))

    return (train, val, test)




def kfold_split(feature_norms: FeatureNorms, embeddings: MultiProtoTypeEmbeddings, k):
    random_seed = 42
    words = list(feature_norms.vocab.keys())
    dataset_size = len(words) 
    print("starting with dataset of size: ", dataset_size)
    size_of_chunk = int(np.floor(dataset_size / k))


    sizes_of_chunks = [size_of_chunk for i in range(k)]


    # the last chunk might be a little bit bigger if the dataset isnt evenlt divisible
    last_chunk_size = dataset_size - (size_of_chunk * (k-1))
    sizes_of_chunks[-1] = last_chunk_size

    #print(sizes_of_chunks)

    chunks = random_split(words, sizes_of_chunks, generator=torch.Generator().manual_seed(random_seed))

    return chunks


def load_feature_norms(args):
    if args['train_data'] == 'mc_rae_real':
        print("gets here")
        feature_norms = McRaeFeatureNorms('./data/external/mcrae/CONCS_FEATS_concstats_brm/concepts_features-Table1.csv')
    elif args['train_data'] == 'mc_rae_subset':
        feature_norms = BuchananFeatureNorms('data/external/buchanan/cue_feature_words.csv', subset='mc_rae_subset')
    elif args['train_data'] == 'buchanan':
        feature_norms = BuchananFeatureNorms('data/external/buchanan/cue_feature_words.csv')
    elif args['train_data'] == 'binder':
        feature_norms = BinderFeatureNorms('data/external/binder_word_ratings/WordSet1_Ratings.csv')
    else:
        raise Exception("dataset not implemented")
    return feature_norms

def load_embeddings(args):
    if args['embedding_type'] == 'bert':
        embedding_file = './data/processed/multipro_embeddings/layer'+ str(args['layer']) + 'clusters' + str(args['clusters']) + '.txt'
        embs = read_multiprototype_embeddings(embedding_file, layer=args['layer'], num_clusters=args['clusters'])

    elif args['embedding_type'] == 'glove':
        embeddings_list = []
        word_indexer = Indexer()
        with open("data/external/glove.6B/glove.6B.300d.txt", 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_list.append([vector])

                #print(embeddings_dict)
                #raise Exception("hfelfnl")
                word_indexer.add_and_get_index(word)

        embs = MultiProtoTypeEmbeddings(word_indexer, np.array(embeddings_list), 0, 1) # dummy layer, clusters = 1

    # zscore normalization based on that one paper
    if args['zscore']:
        print("training on zscore-normalized embeddings")
        reshape = embs.vectors.reshape(61740, 768)
        zscored = stats.zscore(reshape, axis=0)
        rereshape = zscored.reshape(embs.vectors.shape[0], 5, 768)
        embs.vectors = rereshape

    return embs




def train(train_words, dev_words, test_words, embs, feature_norms, args):
    start = time.time()
    if args['do_dumb_thing']:
        model = FrequencyClassifier(feature_norms)
    elif args['model'] == 'binary':
        model = train_binary_classifier(train_words, dev_words, embs, feature_norms, args)
    elif args['model'] == 'ffnn':
        model = train_ffnn(train_words, dev_words, embs, feature_norms, args)
    elif args['model'] == 'label_propagation':
        model = train_label_propagation(train_words, dev_words, embs, feature_norms, args)
    elif args['model'] == 'knn':
        model = train_knn_regressor(train_words, dev_words, embs, feature_norms, args)
    elif args['model'] == 'plsr':
        model = train_plsr(train_words, dev_words, embs, feature_norms, args)
    elif args['model'] == 'modabs':
        model = train_mad(train_words, dev_words, test_words, embs, feature_norms, args)
    else:
        raise Exception("model not implemented: ", args.model)

    end = time.time()
    print("Time elapsed during training: %s seconds" % (end - start))
    return model




def train_1_fold(feature_norms, embs, args):

    train_words, dev_words, test_words = prepare_data(feature_norms, embs, args)

    """
    for debug we might want the dev set to be the trains et just to see if we're learning those
    """
    if args['dev_equals_train']:

        # DEBUG toy dataset
        # train_words = train_words[:10]

        dev_words = train_words

    print("%i train exs, %i dev exs, %i test exs" % (len(train_words), len(dev_words), len(test_words)))


    model = train(train_words, dev_words, test_words, embs, feature_norms, args)

    print("=======DEV SET=======")
    # return (top_10_prec, top_20_prec, top_k_prec, average_correlation, average_cosine)
    results = evaluate(model, dev_words, feature_norms, args, debug='false')
    print(results)
    tune.report(
        dev_MAP_at_10=results['MAP_at_10'],
        dev_MAP_at_20=results['MAP_at_20'],
        dev_MAP_at_k=results['MAP_at_k'],
        dev_correl=results['correl'],
        dev_cos=results['cos'],
        dev_rsquare=results['rsquare'],
        dev_mse=results['mse']
    )

    print("=======FINAL PRINTING ON TEST SET=======")
    # return (top_10_prec, top_20_prec, top_k_prec, average_correlation, average_cosine)
    print("testing on these words:")
    for test_word in test_words:
        print(test_word)
    results = evaluate(model, test_words, feature_norms, args, debug='false')
    tune.report(
        test_MAP_at_10=results['MAP_at_10'],
        test_MAP_at_20=results['MAP_at_20'],
        test_MAP_at_k=results['MAP_at_k'],
        test_correl=results['correl'],
        test_cos=results['cos'],
        test_rsquare=results['rsquare'],
        test_mse=results['mse']
    )

    if (args['save_path'] is not None):
        if not args['tuning']:
            torch.save(model, args['save_path'])
            print("Wrote trained model to ", args['save_path'])   

def kfold_crossvalidation(feature_norms, embs, args):
    k = args["k_fold"]
    dev_results = []

    chunks = kfold_split(feature_norms, embs, k)

    test_index = 0
    train_index = 1

    # split data into 10 equal parts, and then iterate training 10 times.
    dev_stats = []
    test_stats = []
    for i in range(0,k):

        print(" ")
        print(" ")
        print("*------------------------------------------------------------*")
        print("Running k-fold cross validation, k=%s, this is iteration %s" % (k, i))
        # take the first two off and use them
        test_words = chunks.pop(0)
        dev_words = chunks[0]
        # flatten the rest of the list (the k-2 folds) to use as training data. 
        # for k = 10, this takes 8 equal sized lists and makes them one list
        train_words = [item for sublist in chunks[1:] for item in sublist]
        # add the test words back to the end of the list (we took them from the beginning of th elist)
        chunks.append(test_words)

        print("size of train set:", len(train_words))
        print("size of dev set:", len(dev_words))
        print("size of test set:", len(test_words))


        print("%i train exs, %i dev exs, %i test exs" % (len(train_words), len(dev_words), len(test_words)))

        #if args.print_dataset:
            #print("Input indexer: %s" % input_indexer)
            #print("Output indexer: %s" % output_indexer)
            #print("Here are some examples post tokenization and indexing:")
            #for i in range(0, min(len(train_data_indexed), 10)):
            #    print(train_data_indexed[i])   

        model = train(train_words, dev_words, test_words, embs, feature_norms, args)



        # always get dev set results
        results = evaluate(model, dev_words, feature_norms, args, debug='false')
        dev_stats.append(results)

        # if we are doing final eval get test set results
        #if not args['tuning']:
        results = evaluate(model, test_words, feature_norms, args, debug='false')
        test_stats.append(results)


    print("=======DEV SET=======")
    all_folds = pd.DataFrame.from_records(dev_stats)
    print(dev_stats)
    average = all_folds.mean(axis=0)
    tune.report(
        dev_MAP_at_10=average.MAP_at_10,
        dev_MAP_at_20=average.MAP_at_20,
        dev_MAP_at_k=average.MAP_at_k,
        dev_correl=average.correl,
        dev_cos=average.cos,
        dev_rsquare=average.rsquare,
        dev_mse=average.mse
    )
    print(average)

    # if we are doing final eval get test set results
    #if not args['tuning']:
    print("=======FINAL PRINTING ON TEST SET=======")
    all_folds = pd.DataFrame.from_records(test_stats)
    average = all_folds.mean(axis=0)
    tune.report(
        test_MAP_at_10=average.MAP_at_10,
        test_MAP_at_20=average.MAP_at_20,
        test_MAP_at_k=average.MAP_at_k,
        test_correl=average.correl,
        test_cos=average.cos,
        test_rsquare=average.rsquare,
        test_mse=average.mse
    )
    print(average)


def main(args):

    print(args)

    os.chdir(args['TUNE_ORIG_WORKING_DIR'])

    random.seed(args['seed'])
    np.random.seed(args['seed'])

    # Load the training and test data

    print(args['train_data'])

    feature_norms = load_feature_norms(args)
    embs = load_embeddings(args)
 

    if args['k_fold']:
        kfold_crossvalidation(feature_norms, embs, args)
    else:
        train_1_fold(feature_norms, embs, args)


if __name__ == '__main__':
    args = _parse_args()
    args = vars(args)

    args['TUNE_ORIG_WORKING_DIR'] = os.getcwd()
    print(args)

    tune.run(main, config=args)
    #main(args)
