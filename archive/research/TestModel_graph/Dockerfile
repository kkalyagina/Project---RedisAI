FROM ubuntu:latest

LABEL maintainer = "andrei <andrey@savchuk.biz>"

RUN apt update && \
	apt install -y python3-venv iputils-ping && \
	VE="/opt/venv" && \
	python3 -m venv $VE && \
	PATH=$VE/bin:$PATH &&  \
	useradd -ms /bin/bash student && \
	chown student opt && \
	chsh -s /bin/bash student && \
	echo "root:shvG.JsHq1TEY" | chpasswd -e && \
    pip install redisai && \
    pip install ml2rt

#USER	student

#COPY . .

EXPOSE 8886

#WORKDIR ...

CMD export PATH=/opt/venv/bin:$PATH && \
    bash -c 'source /opt/venv/bin/activate' && \
	bash -c 'python3 /data/script.py'
