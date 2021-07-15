#! /bin/sh
docker stop $(docker ps -q)

docker network rm redis_ntw
docker container rm viper
docker container rm toad

docker network create --driver bridge redis_ntw
docker run -p 6379:6379 --name viper --network redis_ntw -d redisai/redisai 
docker image build -t redisai_maskface .
docker run -v $(pwd)/data:/data --name toad --network redis_ntw redisai_maskface


#docker exec -it $(docker ps -q) bash
#docker run -it  -v "$(pwd)/data:/data" --name ubun --network redis_ntw ubuntu:latest bash