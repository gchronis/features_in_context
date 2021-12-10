"""
script to do hyperparameter tuning for ffnn on mcrae data
"""
import subprocess, os


if __name__ == '__main__':

	models = ['plsr']
	#datasets = ['mc_rae_real', 'buchanan']
	datasets = ['binder']
	embeddings = ['5k', '1k', 'glove']
	plsr_n_components = ['50', '100', '300']
	plsr_max_iters = ['500']
	  

	for model in models:
		for dataset in datasets:
			for embedding in embeddings:
				for plsr_n_component in plsr_n_components:
					for plsr_max_iter in plsr_max_iters:
						command = [
							"python3",
							"classifier_main.py",
							"--print_dataset",
							"--model=plsr",
							"--train_data=" + str(dataset),
							'--plsr_n_components=' + str(plsr_n_component) ,
							'--plsr_max_iter=' + str(plsr_max_iter)
							]

						if embedding == '1k':
							command.append("--layer=8 --clusters=1")
						elif embedding =='5k':
							command.append("--layer=8 --clusters=5")
						elif embedding == 'glove':
							command.append("--embedding_type=glove")

						save_path = 'trained_models/model.plsr.' + dataset + '.' + embedding + '.' + plsr_n_component + 'components.' + plsr_max_iter + 'max_iters'
						command.append('--save_path=' + save_path)

						print("running command:")
						print(command)

						os.system(' '.join(command))
