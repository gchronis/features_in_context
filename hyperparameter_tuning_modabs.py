"""
script to do hyperparameter tuning for ffnn on mcrae data
"""
import subprocess, os


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

	  
	  
	# From Python3.7 you can add 
	# keyword argument capture_output
	print(subprocess.run(["echo", "Geeks for geeks"], 
						 capture_output=True))
	  
	# For older versions of Python:
	print(subprocess.check_output(["echo", 
								   "Geeks for geeks"]))
	i = 0
	for model in models:
		for dataset in datasets:
			for embedding in embeddings:
				for mu1 in mu1s:
					for mu2 in mu2s:
						for mu3 in mu3s:
							for mu4 in mu4s:
								for nnk in nnks:
									i +=1
									command = [
										"python3",
										"classifier_main.py",
										"--print_dataset",
										"--model=modabs",
										"--train_data=" + str(dataset),

										'mu1=' + str(mu1) ,
										'mu2=' + str(mu2) ,
										'mu3=' + str(mu3),
										'mu4=' + str(mu4),
										'nnk=' + str(nnk),
										]

									if embedding == '1k':
										command.append("--layer=8 --clusters=1")
									elif embedding =='5k':
										command.append("--layer=8 --clusters=5")
									elif embedding == 'glove':
										command.append("--embedding_type=glove")

									save_path = 'trained_models/model.ffnn.' + dataset + '.' + embedding + '.' + mu1 + 'mu1.' + mu2 + 'mu2.' +  mu3 + 'mu3.' +  mu4 + 'mu4.' + nnk + 'nnk'
									command.append('--save_path=' + save_path)

									print("running command:")
									print(command)
									print(i)
									#os.system(' '.join(command))
