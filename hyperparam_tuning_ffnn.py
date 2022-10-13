"""
script to do hyperparameter tuning for ffnn on mcrae data
"""
import subprocess, os
from ray import tune
import classifier_main
from ray.tune.schedulers import ASHAScheduler


if __name__ == '__main__':

	models = ['ffnn']
	datasets = ['mc_rae_real', 'buchanan', 'binder']
	#datasets = ['binder']

	# uncomment do 1k and glove
	clusters = [1]
	embedding_type = ['bert', 'glove']

	# uncomment to do 5k; avoids repeating glove trials
	#clusters = [5]
	#embedding_type = ['bert']

	epochs = [30, 50]
	dropouts = [0.5, 0.2, 0.0]
	learning_rates = [1e-5, 1e-4, 1e-3]
	hidden_sizes = [50, 100, 300]




	config = {
			"seed": 42,
			"layer": 8,
			"clusters": tune.grid_search(clusters),
			"embedding_type": tune.grid_search(embedding_type),
			"model": tune.grid_search(models),
			"train_data": tune.grid_search(datasets),

			"epochs": tune.choice(epochs),
			"dropout": tune.choice(dropouts),
			"lr": tune.choice(learning_rates),
			"hidden_size": tune.choice(hidden_sizes),
			"batch_size": 1,

		    'TUNE_ORIG_WORKING_DIR': os.getcwd(),
		    "kfold": 10,


		    # BS stuff??
		    "print_dataset": False,
		    "save_path": None,
		    "do_dumb_thing": False,
		    "dev_equals_train": False,
		    "tuning": True,
		    "allbuthomonyms": False
		    "zscore": False

			}


	analysis = tune.run(
		classifier_main.main,
		config=config,
		scheduler=ASHAScheduler(metric="dev_MAP_at_k", mode="max"),
		num_samples=25,
    	#name="main_2022-02-11_15-08-47",
    	name="ffnn_1k_tuning_kfold_10_13_2022",
    	#trial_name_creator = tune.function(lambda trial: trial.config['embedding_type'] + str(trial.config['clusters']) + '_' + trial.trial_id),
    	#resume="AUTO"
    	resume = False
	)

	# Obtain a trial dataframe from all run trials of this `tune.run` call.
	dfs = analysis.trial_dataframes