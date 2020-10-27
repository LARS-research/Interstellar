# Interstellar
The code for our paper ["Interstellar: Searching Recurrent Architecture for Knowledge Graph Embedding"](https://arxiv.org/abs/1911.07132) in NeurIPS 2020.

## Instructions
For the sake of ease, a quick instruction is given for readers to reproduce the searching process.
Note that the programs are tested on Linux (Red Hat 4.8.5-39), Python 3.7.6 from Anaconda 4.8.5

For data packages, please unpack the data.zip file.

Install required packages

    pip install -r requirements

### Interstellar on entity alignment
    
    python -W ignore train_align.py

### Interstellar on link prediction

    python -W ignore train_link.py


