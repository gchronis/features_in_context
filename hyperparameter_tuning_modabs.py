"""
script to do hyperparameter tuning for ffnn on mcrae data
"""
import subprocess, os
import random
import torch.optim as optim
from ray import tune
import classifier_main
from ray.tune.schedulers import ASHAScheduler


# def construct_command(seed=None, model=None, dataset=None, layer=None, clusters=None, embedding=None, mu1=None, mu2=None, mu3=None, mu4=None, nnk=None):

# 	command = [
# 		"python3",
# 		"classifier_main.py",
# 		"--tuning",
# 		"--print_dataset",
# 		"--model=" + model,
# 		"--train_data=" + dataset,

# 		'--mu1=' + str(mu1) ,
# 		'--mu2=' + str(mu2) ,
# 		'--mu3=' + str(mu3),
# 		'--mu4=' + str(mu4),
# 		'--nnk=' + str(nnk),
# 		]

# 	if embedding == '1k':
# 		command.append("--layer=8 --clusters=1")
# 	elif embedding =='5k':
# 		command.append("--layer=8 --clusters=5")
# 	elif embedding == 'glove':
# 		command.append("--embedding_type=glove")

# 	save_path = 'trained_models/model.modabs.' + dataset + '.' + embedding  + '.mu1_' + str(mu1) + '.mu2_' + str(mu2) + '.mu3_' +  str(mu3) + '.mu4_' +  str(mu4) + '.nnk_' + str(nnk)
# 	command.append('--save_path=' + save_path)
# 	return command


if __name__ == '__main__':

	models = ['modabs']
	datasets = ['mc_rae_real']
	#datasets = ['buchanan']
	embeddings = ['5k', '1k', 'glove']
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
			#"clusters": grid_search([1,5]),
			"model": 'modabs',
			"train_data": 'mc_rae_real',
			#"embedding_type": grid_search(['bert', 'glove']),
			"mu1": 1,
			"mu2": tune.choice(mu2s),
		    "mu3": tune.choice(mu3s),
		    "mu4": tune.grid_search(mu4s),
		    "nnk": tune.grid_search(nnks),
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


	# run_experiments({
	#     "my_experiment": {
	#         "run": "train",
	#         "resources": { "cpu": 1, "gpu": 0 },
	#         #"stop": { "mean_accuracy": 100 },
	#         "config": config,
	#     },
	# })

	# run trials for each kind of input embedding, single-prototype BERT, multiprototype BERT, and glove
	input_embedding = [
		('bert', 1),  #1k
		('bert', 5),  #5k
		('glove', 1)  # glove
	]

	for emb in input_embedding:

		# set the parameters for this input embedding
		config['embedding_type'] = emb[0]
		config['clusters'] = emb[1]

		analysis = tune.run(
			classifier_main.main,
			config=config,
			scheduler=ASHAScheduler(metric="MAP_at_k", mode="max"),
			num_samples=15,
	    	#name="main_2022-02-11_15-08-47",
	    	name="modabs_tuning",
	    	trial_name_creator = tune.function(lambda trial: trial.config['embedding_type'] + str(trial.config['clusters']) + '_' + trial.trial_id),
	    	resume=True
		)

	# Obtain a trial dataframe from all run trials of this `tune.run` call.
	dfs = analysis.trial_dataframes

	# for i in range(0, 25):
	# 	print(i)
	# 	command = construct_command(
	# 		# config
	# 		seed = 42,
	# 		layer = 8,
	# 		clusters = 5,
	# 		model = 'modabs',
	# 		dataset = 'mc_rae_real',
	# 		embedding = random.choice(embeddings),
	# 		mu1 = 1,
	# 		mu2 = random.choice(mu2s),
	# 	    mu3 = random.choice(mu3s),
	# 	    mu4 = random.choice(mu4s),
	# 	    nnk = random.choice(nnks)
	# 		)

	# 	print("running command:")
	# 	print(command)
	# 	print(i)
	# 	os.system(' '.join(command))
