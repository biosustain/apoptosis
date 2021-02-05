# Apoptosis resistance in CHO cells

Data and code related to the draft paper "Modelling Classical Apoptosis
Resistance in CHO cells with CRISPR-Mediated Knock-outs of BAK1, BAX, and BOK
Reveals both delayed onset and decreased rates of cell death".

## How to find the analysis

Model assumptions and results can be found in 
[report.md](https://github.com/biosustain/apoptosis/blob/master/report.md) and 
[report.pdf](https://github.com/biosustain/apoptosis/blob/master/report.pdf). 

## How to run the analysis

The following terminal commands should work on any platform provided
[Docker](https://www.docker.com/) is running:

```shell
docker build -t apoptosis_image .

docker run --rm -v $(pwd):/apoptosis apoptosis_image

```

Alternatively, you can use the following instructions.

First install the python dependencies in a python 3 environment:

```shell
pip3 install -r requirements.txt 
```

Next install [cmdstan](https://mc-stan.org/users/interfaces/cmdstan). The
following shell command will work on mac and linux provided there is already a
working C++ toolchain:

```shell
install_cmdstan
```

On windows these powershell commands should work

```shell
python -m cmdstanpy.install_cxx_toolchain
python -m cmdstanpy.install_cmdstan --compiler
```

See
[here](https://cmdstanpy.readthedocs.io/en/v0.9.67/installation.html#install-cmdstan)
for troubleshooting about this way of installing cmdstan.

Finally, run the analysis with the following command:

```shell
python3 run_model.py
```

