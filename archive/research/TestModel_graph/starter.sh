#! /bin/sh
docker stop $(docker ps -q)

docker network rm redis_ntw
docker container rm viper
docker container rm toad

docker network create --driver bridge redis_ntw
docker run -p 6379:6379 --name viper --network redis_ntw -d redisai/redisai 
docker image build -t redisai_study_env .
docker run -v $(pwd)/data:/data --name toad --network redis_ntw redisai_study_env


#docker exec -it $(docker ps -q) bash
