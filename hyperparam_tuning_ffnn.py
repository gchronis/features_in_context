"""
script to do hyperparameter tuning for ffnn on mcrae data
"""
import subprocess, os
from ray import tune
import classifier_main
from ray.tune.schedulers import ASHAScheduler


if __name__ == '__main__':

	models = ['ffnn']
	#datasets = ['mc_rae_real', 'buchanan']
	datasets = ['binder']
	embeddings = ['5k', '1k', 'glove']
	epochs = ['30', '50']
	dropouts = ['0.5', '0.2', '0.0']
	learning_rates = ['1e-5', '1e-4', '1e-3']
	hidden_sizes = ['50', '100', '300']




	config = {
			"seed": 42,
			"layer": 8,
			"clusters": tune.grid_search([1,5]),
			"embedding_type": tune.grid_search(['bert', 'glove']),
			"model": tune.choice(models),
			"train_data": tune.choice(datasets),

			"epochs": tune.choice(epochs),
			"dropout": tune.choice(dropouts),
			"lr": tune.choice(learning_rates),
			"hidden_size": tune.choice(hidden_sizes),
			"batch_size": 1,

		    'TUNE_ORIG_WORKING_DIR': os.getcwd(),

		    # BS stuff??
		    "print_dataset": False,
		    "save_path": None,
		    "do_dumb_thing": False,
		    "kfold": False,
		    "dev_equals_train": False,
		    "tuning": True,
		    "allbuthomonyms": False
			}


	analysis = tune.run(
		classifier_main.main,
		config=config,
		scheduler=ASHAScheduler(metric="MAP_at_k", mode="max"),
		num_samples=25,
    	#name="main_2022-02-11_15-08-47",
    	name="ffnn_tuning_binder",
    	#trial_name_creator = tune.function(lambda trial: trial.config['embedding_type'] + str(trial.config['clusters']) + '_' + trial.trial_id),
    	#resume="AUTO"
    	resume = False
	)

	# Obtain a trial dataframe from all run trials of this `tune.run` call.
	dfs = analysis.trial_dataframes






	  
	# # From Python3.7 you can add 
	# # keyword argument capture_output
	# print(subprocess.run(["echo", "Geeks for geeks"], 
	# 					 capture_output=True))
	  
	# # For older versions of Python:
	# print(subprocess.check_output(["echo", 
	# 							   "Geeks for geeks"]))

	# for model in models:
	# 	for dataset in datasets:
	# 		for embedding in embeddings:
	# 			for epoch in epochs:
	# 				for dropout in dropouts:
	# 					for learning_rate in learning_rates:
	# 						for hidden_size in hidden_sizes:
	# 							command = [
	# 								"python3",
	# 								"classifier_main.py",
	# 								"--print_dataset",
	# 								"--model=ffnn",
	# 								"--train_data=" + str(dataset),
	# 								'--epochs=' + str(epoch) ,
	# 								'--dropout=' + str(dropout),
	# 								'--lr=' + str(learning_rate),
	# 								'--hidden_size=' + str(hidden_size)
	# 								]

	# 							if embedding == '1k':
	# 								command.append("--layer=8 --clusters=1")
	# 							elif embedding =='5k':
	# 								command.append("--layer=8 --clusters=5")
	# 							elif embedding == 'glove':
	# 								command.append("--embedding_type=glove")

	# 							save_path = 'trained_models/model.ffnn.' + dataset + '.' + embedding + '.' + epoch + 'epochs.' + dropout + 'dropout.' + 'lr' + learning_rate + '.hsize' + hidden_size
	# 							command.append('--save_path=' + save_path)

	# 							print("running command:")
	# 							print(command)

	# 							os.system(' '.join(command))
