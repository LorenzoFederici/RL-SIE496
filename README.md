# REINFORCEMENT LEARNING FOR ROBOT CONTROL

This repository is intended for the students of Prof. Curti's Robotic Systems class (SIE 496/596), offered by the Systems & Industrial Engineering Department at the University of Arizona.
The aim of this repository is to show how to formulate a decision-making (or control) problem as a Markov decision process (MDP) by exploiting the OpenAI [Gymnasium](https://gymnasium.farama.org/) Python library, and how to train a deep neural network to solve this problem via a Reinforcement Learning (RL) algorithm using the [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html) Python library.

## INSTALLATION

The software runs on Linux (Ubuntu), macOS, and Windows (through Windows Subsystem for Linux, WSL). 

To correctly set up the software, please follow the present instructions, intended to be run on a Linux terminal:

1. First, you need Python3 and the system packages gcc, g++, and make:
    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install python3-dev build-essential
    ```

2. Then, install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main/) on Linux via command line with the command:
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh
    ```
    Then, press Enter until the installation process is done.

3. Now, open a new terminal and use conda to create a virtual environment, named rl-env, with a specific Python version (3.10 and above) by using conda:
    ```bash
    conda create -n rl-env python=3.10
    ```

3. When the environment is created, activate it with:
    ```bash
    conda activate rl-env
    ```
    and then install all packages required by this project via pip:
    ```bash
    pip install tensorboard swig box2d pygame ale-py gymnasium[all] stable-baselines3
    ```

##  USAGE

- To test an environment with a random agent, specify the environment name within the script `run_env.py`. Then, run the script via:
    ```bash
    python run_env.py
    ```

- To train a new model via RL and save it, specify the environment name, model architecture, algorithm to use, and algorithm hyperparameters within the script `train.py` or `train_atari.py` (for [Atari](https://ale.farama.org/environments/) environments). Then, run the script via:
    ```bash
    python train.py
    ```
    or
    ```bash
    python train_atari.py
    ```

- To check the training progress, you can use TensorBoard by running the following command in the terminal while the training is still running:
    ```bash
    tensorboard --logdir=logs/
    ```
    and then opening the URL `http://localhost:6006/` in your web browser.

- To load a pretrained model, evaluate its performance, and test it in deployment mode, specify the environment name within the script `evaluate.py` or `evaluate_atari.py` (for [Atari](https://ale.farama.org/environments/) environments), and run it via:
    ```bash
    python evaluate.py
    ```
    or
    ```bash
    python evaluate_atari.py
    ```

Enjoy!




