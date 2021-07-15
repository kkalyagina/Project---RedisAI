# To build an image:

docker-compose build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)

# To run containers from the image:
docker-compose up -d

# To print list the running containers:

docker ps -a

# To stop a docker-compose:

docker-compose down

# Make some predictions:

redis-cli AI.TENSORSET predict FLOAT 2 4 VALUES 5.0 3.4 1.6 0.4 6.0 2.2 5.0 1.5

redis-cli AI.MODELRUN onnx_model INPUTS predict OUTPUTS inferences scores

redis-cli AI.TENSORGET inferences VALUES 

