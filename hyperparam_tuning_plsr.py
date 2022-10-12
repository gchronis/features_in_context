"""
script to do hyperparameter tuning for PLSR
"""
import subprocess, os
from ray import tune
import classifier_main
from ray.tune.schedulers import ASHAScheduler


if __name__ == '__main__':

	models = ['plsr']
	#datasets = ['binder', 'mc_rae_real', 'buchanan']
	datasets = ['binder', 'mc_rae_real'] # dont actually tune buchanan it just takes too long; use mcrae params
	embeddings = ['bert', 'glove']
	clusters = [1]
	plsr_n_components = ['30', '50', '100', '300']
	plsr_max_iters = ['500']


	config = {
			"seed": 42,
			"layer": 8,
			"clusters": tune.grid_search(clusters),
			"embedding_type": tune.grid_search(embeddings),
			"model": tune.choice(models),
			"train_data": tune.choice(datasets),

			"plsr_n_components": plsr_n_components
			"plsr_max_iter": plsr_max_iters,	

		    "TUNE_ORIG_WORKING_DIR": os.getcwd(),

		    "kfold": True,

		    # BS stuff??
		    "print_dataset": False,
		    "save_path": None,
		    "do_dumb_thing": False,
		    "dev_equals_train": False,
		    "tuning": True,
		    "allbuthomonyms": False
			}


	run_name = plsr_tuning_kfold_10_11_2022
	analysis = tune.run(
		classifier_main.main,
		config=config,
		scheduler=ASHAScheduler(metric="MAP_at_k", mode="max"),
		
		# DEBUG
		#num_samples=25,
		num_samples = 1,
    	#name="main_2022-02-11_15-08-47",
    	name=run_name,
    	#trial_name_creator = tune.function(lambda trial: trial.config['embedding_type'] + str(trial.config['clusters']) + '_' + trial.trial_id),
    	#resume="AUTO"
    	resume = False
	)

	# Obtain a trial dataframe from all run trials of this `tune.run` call.
	dfs = analysis.trial_dataframes
	dfs.to_csv(name+'.csv')
