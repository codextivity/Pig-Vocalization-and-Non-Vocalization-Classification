
# [DCNN for Pig Vocalization and Non-Vocalization Classification: Evaluate Model Robustness with New Data](https://www.mdpi.com/2076-2615/14/14/2029)

<h3 align="center">
    <a href="https://www.mdpi.com/2076-2615/14/14/2029"><img src="images/1.png" width="100%"></a>
</h3>

## Simple Summary

This study addresses the significance of animal sounds as valuable indicators of both behavior and health in animals, emphasizing the challenges involved in collecting datasets for deep learning models. Particularly, in the context of classifying pig vocalization and non-vocalization, it is identified as laborious and time-consuming when relying on human efforts. In response to these challenges, the research proposes a new approach utilizing a deep learning model to automatically classify pig vocalization and non-vocalization with high accuracy. The success of this method not only provides an efficient means of collecting pig sound datasets but also presents a promising avenue for improving the classification of pig vocalization and non-vocalization in deep learning models, thereby contributing to advancements in animal behavior research and health monitoring.


## What's in this repository?


* Pig-Vocalization-and-Non-Vocalization-Classification
  * [Datasets](https://github.com/codextivity/Pig-Vocalization-and-Non-Vocalization-Classification/tree/main/Datasets): This folder is used to store all dataset.
  * [Utilities](https://github.com/codextivity/Pig-Vocalization-and-Non-Vocalization-Classification/tree/main/Utilities): This folder is used to store files that are necessary for load dataset, dataset augmentation, and calculate audio dBFS (decibels relative to full scale ).
  * [Network](https://github.com/codextivity/Pig-Vocalization-and-Non-Vocalization-Classification/tree/main/network): This folder is used to store backbone network
  * [train.py](https://github.com/codextivity/Pig-Vocalization-and-Non-Vocalization-Classification/blob/main/main.py): This file is used to train the model.
  * [train_with_k_fold.py](https://github.com/codextivity/Pig-Vocalization-and-Non-Vocalization-Classification/blob/main/train_with_k_fold.py): This file is used to train model with k-fold cross-validation technique.
  * [model_robustness.py](https://github.com/codextivity/Pig-Vocalization-and-Non-Vocalization-Classification/blob/main/model_robustness.py): This file is used to find model generalization efficiency by conducting robustness experimentation.

## Installation
#### Install all required libraries with `pip` or `conda` command. 
```
tensorflow=2.10.0
keras==2.10.0
Keras-Preprocessing==1.1.2
librosa=0.10.1
matplotlib=3.7.4
pandas=2.0.3
pydub=0.25.1
numpy==1.24.4
scikit-learn==1.3.2
scipy==1.10.1
seaborn==0.13.0
tensorboard==2.10.1
tqdm==4.66.1

```

## Citations
### BibTeX

```bibtex
@Article{ani14142029,
AUTHOR = {Pann, Vandet and Kwon, Kyeong-seok and Kim, Byeonghyeon and Jang, Dong-Hwa and Kim, Jong-Bok},
TITLE = {DCNN for Pig Vocalization and Non-Vocalization Classification: Evaluate Model Robustness with New Data},
JOURNAL = {Animals},
VOLUME = {14},
YEAR = {2024},
NUMBER = {14},
ARTICLE-NUMBER = {2029},
URL = {https://www.mdpi.com/2076-2615/14/14/2029},
ISSN = {2076-2615},
DOI = {10.3390/ani14142029}
}
```
