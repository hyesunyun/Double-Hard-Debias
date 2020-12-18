## Double-Hard Debias and Null-space Projection: A Comparison and Combination of Recent Approaches to Debias Word Embeddings
[Grace Fan](fan.gr@northeastern.edu), [David M. Liu](liu.davi@northeastern.edu), [Hye Sun Yun](yun.hy@northeastern.edu)

### Reproduced Double-Hard Debias
We worked on reproducing the results of the Double-Hard Debias paper mentioned below.
We had to make some changes to the provided code and also write new code to generate debiased GloVe and Word2Vec word embeddings and also to evaluate them.

We also worked on using the method in the [Null-space projection (INLP) paper](https://github.com/shauli-ravfogel/nullspace_projection) with the Double-Hard pre-processing method. We compared the performance of the word embeddings created from this method with the word embeddings from Double-Hard Debias. [Here](https://github.com/gracefan2020/nullspace_projection) is the forked repository for reproducing INLP.

### Data
We did not include the datasets in this repository due to the large sizes of the datasets.
You can refer to the instructions below to get the original datasets and the datasets from the original paper. 
For word embeddings that we created, please reach out to us directly.

---

## [Double-Hard Debias: Tailoring Word Embeddings for Gender Bias Mitigation](https://arxiv.org/abs/2005.00965)
[Tianlu Wang](http://www.cs.virginia.edu/~tw8cb/), [Xi Victoria Lin](http://victorialin.net/), [Nazneen Fatema Rajani](http://www.nazneenrajani.com/), [Bryan McCann](https://bmccann.github.io/), [Vicente Ordonez](https://www.vicenteordonez.com/), [Caiming Xiong](http://cmxiong.com/)

### Abstract of Original Paper
Word embeddings derived from humangenerated corpora inherit strong gender bias which can be further amplified by downstream models. Some commonly adopted debiasing approaches, including the seminal Hard Debias algorithm (Bolukbasi et al., 2016), apply post-processing procedures that project pre-trained word embeddings into a subspace orthogonal to an inferred gender subspace. We discover that semantic-agnostic corpus regularities such as word frequency captured by the word embeddings negatively impact the performance of these algorithms. We propose a simple but effective technique, Double-Hard Debias, which purifies the word embeddings against such corpus regularities prior to inferring and removing the gender subspace. Experiments on three bias mitigation benchmarks show that our approach preserves the distributional semantics of the pre-trained word embeddings while reducing gender bias to a significantly larger degree than prior approaches.

### Requirements
- Python 3/2
- [Word Embedding Benckmarks](https://github.com/kudkudak/word-embeddings-benchmarks)

### Data
- Word Embeddings: Please download embeddings debiased by our Double-Hard Debias and other word embeddings from ([here](http://www.cs.virginia.edu/~tw8cb/word_embeddings/)) and save them into [data](./data) folder. 
- Special word lists: You can find all word lists used in this project in [data](./data) folder.

### Double-Hard Debias
You can find the detailed steps to implement Double-Hard Debias in [GloVe_Debias](./GloVe_Debias.ipynb)

### Evaluation
1. We evaluated Double-Hard Debias and other debiasing approaches on [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) and [Word2Vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) embeddings. Please check the results in [GloVe_Eval](./GloVe_Eval.ipynb)
and [Word2Vec_Eval](./Word2Vec_Eval.ipynb).
2. If you want to replicate results on coreference systems, we recommend you:
    - Download word embeddings.
    - Refer to [e2r-coref](https://github.com/kentonl/e2e-coref/tree/e2e) about training a end-to-end coreference system.
    - Refer to [WinoBias](https://github.com/uclanlp/corefBias) about using WinoBias dataset to evaluate gender bias in coreference systems.


### Citing
If you find our paper/code useful, please consider citing:

```
@InProceedings{wang2020doublehard,
author={Tianlu Wang and Xi Victoria Lin and Nazneen Fatema Rajani and Bryan McCann and Vicente Ordonez and Caiming Xiong},
title={Double-Hard Debias: Tailoring Word Embeddings for Gender Bias Mitigation},
booktitle = {Association for Computational Linguistics (ACL)},
month = {July},
year = {2020}
}
```

### Kudos
This project is developed based on [gender_bias_lipstick](https://github.com/gonenhila/gender_bias_lipstick) and [word-embeddings-benchmarks](https://github.com/kudkudak/word-embeddings-benchmarks). Thanks for their efforts!
