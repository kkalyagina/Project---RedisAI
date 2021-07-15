FROM ubuntu:latest

LABEL maintainer = "andrei <andrey@savchuk.biz>"

RUN apt update && \
	export DEBIAN_FRONTEND=noninteractive && \
	apt -yq install python3-venv && \
	echo "root:shvG.JsHq1TEY" | chpasswd -e && \
	VE="/opt/venv" && \
	python3 -m venv $VE && \
	PATH=$VE/bin:$PATH && \
	bash -c 'source /opt/venv/bin/activate' &&\
	pip install mlflow==1.14.1 psycopg2-binary==2.8.6 boto3==1.17.46

EXPOSE 5000

CMD /bin/bash ./mlflow_image/start_mlflow_srv.sh