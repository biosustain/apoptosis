FROM ubuntu:latest

RUN apt-get update && apt-get install -y build-essential python3.7 python3-pip

COPY requirements.txt apoptosis/

RUN pip3 install -r apoptosis/requirements.txt

RUN install_cmdstan -v 2.26.0

COPY . apoptosis/

CMD cd apoptosis && make clean_stan && python3 run_model.py
