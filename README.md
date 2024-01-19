# Unsupervised Open Relation Extraction via Large-Small Model Interaction

### Overview

This repo contains the code and datasets for "Unsupervised Open Relation Extraction via Large-Small Model Interaction".

The framework of our proposed model InstructORE is depicted below:

![framework](framewor.jpg)


### Data

We used two typical datasets, Fewrel and Tacred.

The overall data folder structure is as follows: 

```
data
 ├─Fewrel
    ├──train.json
    ├──test.json
 ├─Tacred
    ├──train.json
    ├──test.json   
```

Download the pre-trained bert-base-uncased via this link (https://huggingface.co/bert-base-uncased)， then put it under the folder named "model".

You also need to prepare a GPT-4 API account.



### Requirements

This repository is tested on pytorch 1.13.0. Please run

```
pip install -r requirements.txt
```

to install all dependencies.



### Code Usage

Run Model InstructORE.

```
# You need to ensure that the file config.yaml is correct
python run.py
```

