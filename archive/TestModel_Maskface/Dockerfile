FROM ubuntu:latest

LABEL maintainer = "andrei <andrey@savchuk.biz>"

COPY data/requirements.txt data/requirements.txt

RUN apt update && \
	export DEBIAN_FRONTEND=noninteractive && \
	apt -yq install python3-venv iputils-ping libgl1-mesa-glx libgtk2.0-dev && \
	echo "root:shvG.JsHq1TEY" | chpasswd -e && \
	VE="/opt/venv" && \
	python3 -m venv $VE && \
	PATH=$VE/bin:$PATH && \
	bash -c 'source /opt/venv/bin/activate' &&\
	pip install -r data/requirements.txt
#	useradd -ms /bin/bash student && \
#	chown student opt && \
#	chsh -s /bin/bash student && \
#	echo "root:shvG.JsHq1TEY" | chpasswd -e && \


#USER	student

#COPY . .

EXPOSE 8886

#WORKDIR ...

CMD export PATH=/opt/venv/bin:$PATH && \
    bash -c 'source /opt/venv/bin/activate' && \
	bash -c 'python3 /data/script.py'
