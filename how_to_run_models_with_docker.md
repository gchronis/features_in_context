## Getting started with docker


First install Docker Desktop: https://www.docker.com/products/docker-desktop/ 

This might require you to update your computer. I had to update my Mac OS to the latest version.

You may be prompeted to make an account and link it to your email. 


We will use an existing basic docker container that is designed to support jupyter notebooks. This will allow us all to work in an identical environment. 

I used this page for instructions: https://towardsdatascience.com/how-to-run-jupyter-notebook-on-docker-7c9748ed209f
It also has a good tutorial of basic docker usage, including how to clean up containers that you are no longer using.

## Running a Jupyter Notebook in a Docker container


We will first see if we can successfully download and run the minimal-notebook container

`$ docker run -p 8888:8888 jupyter/minimal-notebook `

This should download the container image automatically and start it on your local machine.

You can control-click on either of the links in the bottom of the output to open the Jupyter Lab interface in your browser.


## Connecting the local directory to a Docker container

Next step is to make it so our docker container can see the features_in_context repository and the feature prediction model. From last time, you should have a repo on your computer called `features_in_context`

cd path/to/features_in_context

git fetch # pullslatest changes from remote github repository
git merge --ff-only origin/master # fast forwards the remote changes over your local branch

Now, make sure that the subdirectory `trained_models` contains the model files. (Available on request from Author)


```$ ls trained_models
model.plsr.buchanan.allbuthomonyms.5k.300components.500max_iters
model.ffnn.binder.5k.50epochs.0.5dropout.lr1e-4.hsize300
```

Now we want to start our container so that it can see our code and model files. 

Docker volumes are directories (or files) that are outside of the default Docker file system and exist as normal directories and files on the host filesystem. A volume does not increase the size of the containers using it, and the volume’s contents exist outside the lifecycle of a given container.
We can create a volume using -v option.

The following command will start the docker container with a mounted volume containing the files in the features_in_context directory


`$ docker run -p 8888:8888 -v /Users/yourname/path/to/features_in_context:/home/jovyan/work jupyter/minimal-notebook`

If you want to use the current working directory, use $(pwd).

`$ docker run -p 8888:8888 -v $(pwd):/home/jovyan/work jupyter/minimal-notebook`


Inside docker there is a Launcher that allows you to run a terminal, open and edit jupyter notebooks in our mounted Volume

Open the terminal and run

```$ pip install -r `requirements.txt` ```

Now on the left menu use the file tree to navigate to `notebooks/examine_features_in_context.ipynb`

Double click on it to open it. 

Cross your fingers and hope that it works!


TODO: if this feels good an intuitive to use, I may make a custom docker image that has the code and models in it, so it will be possible to just download off the internet and run, without necessarily connecting to local files.