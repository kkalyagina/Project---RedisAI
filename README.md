# RedisAI

Project for testing Redis AI.

Team Lead: Andrei Gavrilov

Data Scientists: Konstantin Sviblov, Kristina Kaliagina, Andrei Savchuk


**START instruction:**

Default OS - Ubuntu on VM
To run on MasOX - comment line 22 "addgroup --gid $GROUP_ID mask && \" in ./Dockerfiles/flask_image/Dockerfile
To run on Windows - don't. Run VM with linux and run in it

1. Fix a docker. (optional)

`sudo chmod 666 /var/run/docker.sock`

2. To build an image.

`sudo docker-compose build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)`

3. To run an image

`docker-compose run -e .env training bash`

or

`docker-compose --env-file .env up`

`docker exec -it training_cont bash`

mlflow UI - https://localhost:5000

4. To run training process 

!!! WARNING !!! training is not working without GPU for now. Made a bypass (loading from file with made-up metrics). 

To test with bypass:

:/app# `python3 scripts/launch_fake_training.py --dataset=small`

To start actual training :

download small CMFD + IMFD dataset for the faster training from [here](https://drive.google.com/file/d/1uDOrFCAq6QuwLNjZvAFFJWM41aCFN-T8/view)
and move it to ./data/datasets/Masks-small

:/app# `python3 scripts/launch_training.py --dataset=small`

5. Request for Flask.

`curl -X POST -F image=@/$(pwd)/data/images/44948_Mask.jpg 'http://0.0.0.0:5555/predict'`

`curl -X POST -F image=@/$(pwd)/data/images/image.jpeg 'http://0.0.0.0:5555/predict'`

6. To stop a docker-compose.

`docker-compose down`



**Full training dataset**

1. Download the dataset.

Download CMFD + IMFD dataset from [here](https://github.com/cabani/MaskedFace-Net#dataset)

Download LFW dataset from [here](https://drive.google.com/u/0/uc?export=download&confirm=rGPs&id=1kJN_ehUpzZaQNqA5YS0ggAnVZ-3Kl8Eq)

2. Unzip both to the same folder.
 

**Output.**
 
 You can find 2 files in ./output folder. There is two models trained by TensorFlow 1.15. First model is TF1.15 .pb graph, and second model is converted TFLite model.

**Description of docker-compose.yaml:**

- run a mlflow-service
- run a server RedisAI
- run Flask app, it loads model as file and set it to mlflow and redis
- run training container (empty default command, to run training use docker exec and run scripts/launch_training.py)

**change tracking**

LAST CHANGES (ANDREI SAVCHUK, April 5-14th, 2021) :
1. Renamed Dockerfile_for_dc to Dockerfile (dockerfile wasn't used in any way anyway)
2. Edited Dockerfiles a bit to bestpractices (less RUNs), added missing "COPY Setup.py . "
3. Cleaned requirements for flask image (which doesn't need no keras nor tensorflow)
4. Moved full training requirements (from Konstantine) to ./Dockerfiles/training
5. Added postgres container
6. Implemented RedMask.modelserve 
7. all dockerfiles in ./Dockerfiles/
8. change model to TFLite version (Redis RAM error on TF 1.15 me dunno why)
9. Removed to archive ./RedMask/tracking (redundand scripts)
10. Removed RedMask/utils/redisai_func


WHAT SHOULD BE DONE:
1. Cleaned the sht out of requirements file for teh RedMask module (./Dockerfile/)
2. remove scripts/tensors.py
4. make flask_servise.py use utils/tensors_func.py
5. Make training work on CPU
6. Add func to select best model from mlflow based on accuracy metric
7. Split ModelLoader to 2 classes (for MLflow and RedisAI interactions)

**keep in mind**

S3 bucket used in this project is my personal bucket, so plz do not exceed my free space or I will be charged and forced to sell my kidney. And I only have 3 of them left.

**Prospector**

Prospector is a tool to analyse Python code and output information about errors, potential problems, convention violations and complexity.

`prospector --profile profile.yaml | tee /$(pwd)/prospect_errors.txt`
