
## [DCNN for Pig Vocalization and Non-Vocalization Classification: Evaluate Model Robustness with New Data](https://www.mdpi.com/2076-2615/14/14/2029)


### Simple Summary

____
This study addresses the significance of animal sounds as valuable indicators of both behavior and health in animals, emphasizing the challenges involved in collecting datasets for deep learning models. Particularly, in the context of classifying pig vocalization and non-vocalization, it is identified as laborious and time-consuming when relying on human efforts. In response to these challenges, the research proposes a new approach utilizing a deep learning model to automatically classify pig vocalization and non-vocalization with high accuracy. The success of this method not only provides an efficient means of collecting pig sound datasets but also presents a promising avenue for improving the classification of pig vocalization and non-vocalization in deep learning models, thereby contributing to advancements in animal behavior research and health monitoring.


### What's in this repository?

---

* Pig-Vocalization-and-Non-Vocalization-Classification
  * [Datasets](https://github.com/codextivity/Pig-Vocalization-and-Non-Vocalization-Classification/tree/main/Datasets): This folder is used to store all dataset.
  * [Utilities](https://github.com/codextivity/Pig-Vocalization-and-Non-Vocalization-Classification/tree/main/Utilities): This folder is used to store files that are necessary for load dataset, dataset augmentation, and calculate audio dBFS (decibels relative to full scale ).
  * [Network](https://github.com/codextivity/Pig-Vocalization-and-Non-Vocalization-Classification/tree/main/network): This folder is used to store backbone network
  * [train.py](https://github.com/codextivity/Pig-Vocalization-and-Non-Vocalization-Classification/blob/main/main.py): This file is used to train the model.
  * [train_with_k_fold.py](https://github.com/codextivity/Pig-Vocalization-and-Non-Vocalization-Classification/blob/main/train_with_k_fold.py): This file is used to train model with k-fold cross-validation technique.
  * [model_robustness.py](https://github.com/codextivity/Pig-Vocalization-and-Non-Vocalization-Classification/blob/main/model_robustness.py): This file is used to find model generalization efficiency by conducting robustness experimentation.

### Installation

---

### Citations

---

### Licensing

---