"""
script to do hyperparameter tuning for ffnn on mcrae data
"""
import subprocess, os


if __name__ == '__main__':

	models = ['ffnn']
	#datasets = ['mc_rae_real']
	datasets = ['buchanan']
	embeddings = ['5k', '1k', 'glove']
	epochs = ['30', '50']
	dropouts = ['0.5', '0.2', '0.0']
	learning_rates = ['1e-5', '1e-4', '1e-3']
	hidden_sizes = ['50', '100', '300']

	  
	  
	# From Python3.7 you can add 
	# keyword argument capture_output
	print(subprocess.run(["echo", "Geeks for geeks"], 
						 capture_output=True))
	  
	# For older versions of Python:
	print(subprocess.check_output(["echo", 
								   "Geeks for geeks"]))

	for model in models:
		for dataset in datasets:
			for embedding in embeddings:
				for epoch in epochs:
					for dropout in dropouts:
						for learning_rate in learning_rates:
							for hidden_size in hidden_sizes:
								command = [
									"python3",
									"classifier_main.py",
									"--print_dataset",
									"--model=ffnn",
									"--train_data=" + str(dataset),
									'--epochs=' + str(epoch) ,
									'--dropout=' + str(dropout),
									'--lr=' + str(learning_rate),
									'--hidden_size=' + str(hidden_size)
									]

								if embedding == '1k':
									command.append("--layer=8 --clusters=1")
								elif embedding =='5k':
									command.append("--layer=8 --clusters=5")
								elif embedding == 'glove':
									command.append("--embedding_type=glove")

								save_path = 'trained_models/model.ffnn.' + dataset + '.' + embedding + '.' + epoch + 'epochs.' + dropout + 'dropout.' + 'lr' + learning_rate + '.hsize' + hidden_size
								command.append('--save_path=' + save_path)

								print("running command:")
								print(command)

								os.system(' '.join(command))
