"""
script to do hyperparameter tuning for ffnn on mcrae data
"""
import subprocess, os
import random
import torch.optim as optim
from ray import tune
from classifier_main import main


def construct_command(seed=None, model=None, dataset=None, layer=None, clusters=None, embedding=None, mu1=None, mu2=None, mu3=None, mu4=None, nnk=None):

	command = [
		"python3",
		"classifier_main.py",
		"--tuning",
		"--print_dataset",
		"--model=" + model,
		"--train_data=" + dataset,

		'--mu1=' + str(mu1) ,
		'--mu2=' + str(mu2) ,
		'--mu3=' + str(mu3),
		'--mu4=' + str(mu4),
		'--nnk=' + str(nnk),
		]

	if embedding == '1k':
		command.append("--layer=8 --clusters=1")
	elif embedding =='5k':
		command.append("--layer=8 --clusters=5")
	elif embedding == 'glove':
		command.append("--embedding_type=glove")

	save_path = 'trained_models/model.modabs.' + dataset + '.' + embedding  + '.mu1_' + str(mu1) + '.mu2_' + str(mu2) + '.mu3_' +  str(mu3) + '.mu4_' +  str(mu4) + '.nnk_' + str(nnk)
	command.append('--save_path=' + save_path)
	return command


if __name__ == '__main__':

	models = ['modabs']
	datasets = ['mc_rae_real']
	#datasets = ['buchanan']
	embeddings = ['5k', '1k', 'glove']
	mu1s = ["1"]
	mu2s = [str(m) for m in [10e-8, 10e-4,10e-2, 1, 10, 100, 1000]]
	mu3s = [str(m) for m in [10e-8, 10e-4,10e-2, 1, 10, 100, 1000]]
	mu4s = [str(m) for m in [1,5,10,20]]
	nnks = [str(n) for n in [3,4,5,6]]


	for i in range(0, 25):


		print(i)
		command = construct_command(
			# config
			seed = 42,
			layer = 8,
			clusters = 5,
			model = 'modabs',
			dataset = 'mc_rae_real',
			embedding = random.choice(embeddings),
			mu1 = 1,
			mu2 = random.choice(mu2s),
		    mu3 = random.choice(mu3s),
		    mu4 = random.choice(mu4s),
		    nnk = random.choice(nnks)
			)

		print("running command:")
		print(command)
		print(i)
		os.system(' '.join(command))
