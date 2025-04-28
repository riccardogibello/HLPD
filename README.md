# Decoding the Hierarchy: A Hybrid Approach to Hierarchical Multi-Label Text Classification


## Introduction


![Alt text](./model_architecture.svg "HLDP architecture. The  Text Encoder produces  text embeddings. The Label Encoder embeds description of the nodes. HLP Decoder applies Hierarchical Self-Attention to the label embeddings using the Hierarchical Mask  and performs Cross-Attention between the labels and the text (c.f. Fig. \ref{fig:decoder}). This process generates a text-wise and label-wise representation, on top of which classification headers are applied. Green nodes denote correct
labels.")


Hierarchical multi-label text classification (HMTC) aims to predict multiple labels from a tree-like hierarchy for a given input text. Recent approaches frame HMTC as a Seq2Seq problem, where the objective is to predict the sequence of associated labels, regardless of their order or position in the hierarchy. Despite promising results, these approaches rely solely on attention mechanisms from previously generated tokens. This limit prevents them from acquiring information about the global hierarchy and may lead to the accumulation of errors as the model learns hierarchical cues among labels.
We propose a novel HMTC model based on a hybrid version of the encoder-decoder architecture where the decoder is pre-populated with the entire label embeddings. By leveraging the decoder's cross-attention and hierarchical self-attention mechanisms, we achieve a label representation that benefits from instance and global label-wise information.
Empirical experiments on four HMTC benchmark datasets demonstrated the effectiveness of our approach by settling new state-of-the-art results. Code and datasets are made available to facilitate the reproducibility and future work.



## Datasets
We conduct experiments on four public datasets:
- Reuters corpus [RCV1-V2](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm)
- Blurb Genre Collection [BGC](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html)
- Web-Of-Science [WOS](https://data.mendeley.com/datasets/9rw3vkcfy4/2)
- AAPD

The original Reuters corpus dataset can be acquired by signing an agreement.
You can find the other datasets in the data repository: data/hiera_multilabel_bench/data_files. 

## Experiments

To run experiments please use the `train_DATASET_NAME.sh` shell script.


## Main Requirements

- torch==1.12.0
- transformers==4.20.0
- datasets==2.6.1
- scikit-learn==1.0.0
- tqdm==4.62.0
- wandb==0.12.0


## Citation


- Torba, F., Gravier, C., Laclau, C., Kammoun, A., Subercaze, J. (2025). Decoding the Hierarchy: A Hybrid Approach to Hierarchical Multi-label Text Classification. In: Hauff, C., et al. Advances in Information Retrieval. ECIR 2025. Lecture Notes in Computer Science, vol 15572. Springer, Cham. https://doi.org/10.1007/978-3-031-88708-6_26

  
- Code inspired from : [*Yova Kementchedjhieva and Ilias Chalkidis. 2023. An Exploration of Encoder-Decoder Approaches to Multi-Label Classification for Legal and Biomedical Text. In Findings of the Association for Computational Linguistics: ACL 2023, pages 5828–5843, Toronto, Canada. Association for Computational Linguistics.*](https://aclanthology.org/2023.findings-acl.360/)
```
@inproceedings{kementchedjhieva-chalkidis-2023-exploration,
    title = "An Exploration of Encoder-Decoder Approaches to Multi-Label Classification for Legal and Biomedical Text",
    author = "Kementchedjhieva, Yova  and
      Chalkidis, Ilias",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.360",
    pages = "5828--5843"
}
```
