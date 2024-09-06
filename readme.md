This repository contains the data and code for the LimGAN algorithm.

Install
Install dependencies with:
```
pip install -r requirements.txt
```

Datasets
Datasets used in this study is CIFAR10, as shown in the "data" directory.

Code
The core code consists of two parts, as shown in the "code" directory:

1. Train a Generative Adversiral Network:
   - Script:
     - en_GAN.py

2. MCMC Sampling:
   - Scripts:
     - MCMC.py
     - en_repgan.py
     - calibration.py
     - dcgan.py


Execution Commands:
To reproduce the results of LimGAN, execute the following commands sequentially:
```
python en_GAN.py
python MCMC.py
```
