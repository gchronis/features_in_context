"""
script to do hyperparameter tuning for ffnn on mcrae data
"""
import subprocess, os
import random
import torch.optim as optim
from ray import tune
import classifier_main
from ray.tune.schedulers import ASHAScheduler

if __name__ == '__main__':

	models = ['modabs']
	datasets = ['binder', 'mc_rae_real', 'buchanan']
	#datasets = ['mc_rae_real'] # THIS IS DONE
	#datasets = ['buchanan'] # this is NOT done

	#embeddings = ['bert', 'glove']
	#clusters = [1]

	embeddings = ['bert']
	clusters = [5]


	mu1s = [1]
	mu2s = [10e-8, 10e-4,10e-2, 1, 10, 100, 1000]
	mu3s = [10e-8, 10e-4,10e-2, 1, 10, 100, 1000]
	mu4s = [1,5,10,20]
	nnks = [3,4,5,6]



	# def my_func(config, reporter):
	#     import time, numpy as np
	#     i = 0
	#     while True:
	#         reporter(timesteps_total=i, mean_accuracy=i ** config["alpha"])
	#         i += config["beta"]
	#         time.sleep(.01)

	#register_trainable("train", classifier_main.main)


	config = {
			"seed": 42,
			"layer": 8,
			"clusters": tune.grid_search(clusters),
			"embedding_type": tune.grid_search(embeddings),
			"model": tune.grid_search(models),
			"train_data": tune.grid_search(datasets),

			"mu1": 1,
			"mu2": tune.choice(mu2s),
		    "mu3": tune.choice(mu3s),
		    "mu4": tune.choice(mu4s), #should grid search
		    "nnk": tune.choice(nnks), # should grid search

		    'TUNE_ORIG_WORKING_DIR': os.getcwd(),
		    "k_fold": 10,


		    # BS stuff??
		    "print_dataset": False,
		    "save_path": None,
		    "do_dumb_thing": False,
		    "dev_equals_train": False,
		    "tuning": True,
		    "allbuthomonyms": False,
		    "zscore": False
			}


	run_name = 'modabs_5k_tuning_kfold_10_13_2022'
	analysis = tune.run(
		classifier_main.main,
		config=config,
		scheduler=ASHAScheduler(metric="dev_MAP_at_k", mode="max"),
		#DEBUG
		num_samples=25,
		#num_samples=1,
    	
    	#name="main_2022-02-11_15-08-47",
    	name=run_name,
    	#trial_name_creator = tune.function(lambda trial: trial.config['embedding_type'] + str(trial.config['clusters']) + '_' + trial.trial_id),
    	#resume="AUTO"
    	resume = "AUTO"
	)

	# Obtain a trial dataframe from all run trials of this `tune.run` call.
	dfs = analysis.trial_dataframes