# Features in Context

This repo contains code for predicting semantic features in context.

First, clone the repo over ssh:

`git clone git@github.com:gchronis/features_in_context.git`

## Using a saved model to predict features

1. obtain a model save file, which is named something like `model.plsr.buchanan.allbuthomonyms.5k.300components.500max_iters`

2. make a directory in the top level of the repo called `trained_models`

`mkdir trained_models`

3. move the save file to this directory

`mv <model save file> ./trained_models`

4. navigate to the notebooks directory

`cd notebooks`

4. launch jupyter

`jupyter notebook`

5. open `examine_features_in_context`


## Training a model from scratch

To train a model from scratch use the script `classifier_main.py`, e.g.

`python3 classifier_main.py --model=plsr --allbuthomonyms --k_fold=4 --layer=8 --clusters=1 --embedding_type=glove --train_data=mc_rae_real --plsr_n_components=100 --plsr_max_iter=500`


or

`python3 classifier_main.py --train_data=buchanan --allbuthomonyms --embedding_type=glove --model=ffnn --layer=8 --clusters=1 --epochs=50 --dropout=0.0 --lr=1e-4 --hidden_size=300 --save_path='trained_models/model.ffnn.buchanan.glove.50epochs.0.0dropout.lr1e-4.hsize300'`

or

`python3 classifier_main.py --train_data=binder --seed=42 --embedding_type=bert   --model=modabs  --layer=8 --clusters=5 --mu1=1 --mu2=0.1  --mu3=1e-07 --mu4=5 --nnk=3 --save_path='trained_models/main_82b2e_00003_3_clusters=5,embedding_type=bert,model=modabs,mu2=0.1,mu3=1e-07,mu4=5,nnk=3,train_data=binder_2022-10-13_20-50-01'`



### Non-optional arguments
```

| --train_data=x     | x can be mc_rae_real, buchanan, binder (also implemented mcrae, which uses lemmatized normalized buchanan version of mcrae norms) |
| --model=x          | `plsr` or `ffnn` or `modabs` (a.k.a. label propagation) |
| --embedding_type=x | `glove` or `bert` |
| --layer=x          | layer of bert embedding to use (we have embeddings for layer 8 and 11 atm, but can make more) |
| --clusters=x       | number of clusters in multiprototype embeddings (1 or 5; if using glove, this is always 1) |
```

### Other options
```
| --k-fold=n  | Do k-fold crossvalidation with n folds. reports average metrics over all folds |
| --allbuthomonyms | trains on all words in the test set except for  |
```

### Model-specific arguments

PLSR
| --plsr_n_components=x | |
| --plsr_max_iter=x | |

FFNN
| --epochs=n |  integer number of training epochs |
| --dropout=n  | dropout (float between 0 and 1, e.g. 0.2) | 
| --lr=1e-4 	| learning rate |
| --hidden_size=300 | number of weights for hidden layers |

Label Propagation 
| --mu1=1 | - |
| --mu2=0.1 | - |
| --mu3=1e-07 | - |
| --mu4=5 | - |
| --nnk=3 | - |
