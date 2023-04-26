# Locally Corrected Nyström  (LCN)

Reference implementation of the locally corrected Nyström kernel approximation, sparse Sinkhorn and LCN-Sinkhorn. These fast methods for optimal transport have been proposed in our paper 

**[Scalable Optimal Transport in High Dimensions for Graph Distances, Embedding Alignment, and More](https://www.cs.cit.tum.de/daml/lcn/)**   
by Johannes Gasteiger, Marten Lienen, Stephan Günnemann  
Published at ICML 2021.

Note that the author's name has changed from Johannes Klicpera to Johannes Gasteiger.

The paper furthermore proposed the graph transport network (GTN), whose implementation you can find in [this accompanying repository](https://github.com/gasteigerjo/gtn).

In addition to LCN and sparse Sinkhorn, this repository furthermore contains useful implementations of various other variants of Sinkhorn:
- full Sinkhorn
- unbalanced Sinkhorn
- learnable unbalanced Sinkhorn via BP-matrix
- Nyström-Sinkhorn (regular and BP version)
- Multiscale Sinkhorn (regular and BP version)
- sparse Sinkhorn (regular and BP version)
- LCN-Sinkhorn (regular and BP version)

All implementations are batched and executable on CPU or GPU via PyTorch. Most algorithms can calculate the distance or the full transport plan (denoted with an `arg_` prefix). Implementations are currently limited to uniform marginals. Backpropagation of Sinkhorn distances are accelerated using analytical gradients. The Nyström kernel approximation can be combined with any sparse approximation to obtain an LCN approximation.

The repository furthermore contains multiple methods for finding landmarks (as used in Nyström) and clusters (for sparse and multiscale Sinkhorn):
- k-means
- hierarchical k-means
- k-means++ and uniform sampling
- angular (cross-polytope) LSH

## Installation
You can install the repository using `pip install -e .`.

## Word embedding data
You can obtain the word embeddings from the [fastText project page](https://fasttext.cc/) and the bilingual lexicons from the [MUSE project page](https://github.com/facebookresearch/MUSE). You can do this directly using the following bash code:
```bash
mkdir data
cd data

# Download word embeddings
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.es.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.de.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fr.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ru.vec

# Download lexica
lgs="de es fr ru"
for lg in ${lgs}
do
  for suffix in .0-5000.txt .5000-6500.txt
  do
    fname=en-$lg$suffix
    wget https://dl.fbaipublicfiles.com/arrival/dictionaries/$fname
    fname=$lg-en$suffix
    wget https://dl.fbaipublicfiles.com/arrival/dictionaries/$fname
  done
done
```

## Running embedding alignment
This repository contains a notebook for running word embedding alignment (`embedding_alignment.ipynb`) and a script for running this on a cluster with [SEML](https://github.com/TUM-DAML/seml) (`embedding_alignment_seml.py`).

The config file specifies all hyperparameters and allows reproducing the results in the paper.

## Contact
Please contact j.gasteiger@in.tum.de if you have any questions.

## Cite
Please cite our paper if you use our method or code in your own work:

```
@inproceedings{gasteiger_2021_lcn,
  title={Scalable Optimal Transport in High Dimensions for Graph Distances, Embedding Alignment, and More},
  author={Gasteiger, Johannes and Lienen, Marten and G{\"u}nnemann, Stephan},
  booktitle = {Thirty-eighth International Conference on Machine Learning (ICML)},
  year={2021},
}
```
