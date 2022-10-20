"""
script to do hyperparameter tuning for PLSR
"""
import subprocess, os
import ray
from ray import tune
import classifier_main
from ray.tune.schedulers import ASHAScheduler


if __name__ == '__main__':
	# limite object storage to 8gb ram
	#num_bytes = (10**10)
	#ray.init(object_store_memory=num_bytes)

	models = ['plsr']
	#datasets = ['binder', 'mc_rae_real', 'buchanan']
	datasets = ['binder', 'mc_rae_real'] # dont actually tune buchanan it just takes too long; use mcrae params
	#embeddings = ['bert', 'glove']
	embeddings = ['bert'] # no glove embeddings for 5k
	clusters = [5]
	plsr_n_components = [30, 50, 100, 300]
	plsr_max_iters = [500]


	config = {
			"seed": 42,
			"layer": 8,
			"clusters": tune.grid_search(clusters),
			"embedding_type": tune.grid_search(embeddings),
			"model": tune.grid_search(models),
			"train_data": tune.grid_search(datasets),

			"plsr_n_components": tune.grid_search(plsr_n_components),
			"plsr_max_iter": tune.choice(plsr_max_iters),	

		    "TUNE_ORIG_WORKING_DIR": os.getcwd(),

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


	run_name = 'plsr_tuning_kfold_10_11_2022'
	analysis = tune.run(
		classifier_main.main,
		config=config,
		scheduler=ASHAScheduler(metric="dev_MAP_at_k", mode="max"),
		
		# DEBUG
		#num_samples=25,
		#num_samples = 4,
    	name=run_name,
    	#trial_name_creator = tune.function(lambda trial: trial.config['embedding_type'] + str(trial.config['clusters']) + '_' + trial.trial_id),
    	#resume="AUTO"
    	resume = False
	)

	# Obtain a trial dataframe from all run trials of this `tune.run` call.
	dfs = analysis.trial_dataframes
	dfs.to_csv(name+'.csv')
