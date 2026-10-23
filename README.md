# Learning Domain- and Class-Disentangled Prototypes for Domain-Generalized EEG Emotion Recognition (TAFFC, 2026)
###   Guangli Li, [Canbiao Wu](https://github.com/WuCB-BCI "") (wucanbiao@m.scnu.edu.cn), Zhehao Zhou, Na Tian, Pengwen Dai, Yuhao Wang, Li Zhang* , and Zhen Liang*
*   This repository contains the implementation of the paper "Learning Domain- and Class-Disentangled Prototypes for Domain-Generalized EEG Emotion Recognition," accepted in IEEE TAFFC 2026.
*   Get the article: [arXiv](https://arxiv.org/pdf/2509.01135? "") or [IEEE](https://ieeexplore.ieee.org/document/11714476 "")

# Abstract
Electroencephalography (EEG)-based emotion recognition plays a critical role in affective Brain–Computer Interfaces (aBCIs), yet its practical deployment remains limited by inter-subject variability, reliance on target-domain data, and unavoidable label noise. To address these challenges, we propose a **M**ulti-domain **A**ggregation **T**ransfer learning framework with domain–class prototypes (MAT) for emotion recognition under completely unseen target domains. MAT introduces a feature decoupling module to disentangle class-invariant domain features from domain-invariant class features, enabling more robust and interpretable EEG representations. A Hierarchical-Domain Aggregation (HDA) mechanism based on Maximum Mean Discrepancy (MMD) constructs superdomains to model shared distributional structures across subjects, while adaptive prototype updating refines domain and class prototypes to capture stable intrinsic representations. Moreover, a pairwise learning strategy reformulates classification as similarity estimation between sample pairs, effectively mitigating the effect of label noise. We conduct extensive experiments on four public EEG emotion datasets (SEED, SEED-IV, SEED-V, and SEED-VII). The results show that the accuracy of MAT is improved by 2.78%, 3.84%, 4.45%, and 2.21% compared with the state-of-the-art (SOTA) unseen-target models. Furthermore, the proposed framework generalizes effectively to clinical depression identification scenarios, validating its application potential in real-world settings. The major contributions of this work are summarized as follows：
*   We propose a unified framework (MAT) that disentangles domain and class representations, conceptualizing inter-subject EEG variability as a feature shift between domain-invariant and class-invariant components.
*   We design a HDA mechanism with adaptive prototype updating to learn domain and class prototypes, effectively capturing shared cross-domain structures and enhancing generalization to unseen subjects.
*   We conduct extensive cross-subject validation on four public EEG datasets. Despite the absence of targetdomain data, MAT outperforms or matches state-ofthe-art transfer learning models and has been successfully extended to clinical mental illness detection scenarios.

# Installation
*   Python 3.8
*   Pytorch 2.0.0
*   NVIDIA CUDA 11.8
*   NVIDIA CUDNN 8700
*   Numpy 1.24.3
*   Scikit-learn 0.22.1
*   scipy 1.5.2 
*   GPU NVIDA GeForce RTX 3090 / 1080Ti / 4090
# Databases
*   [SEED](https://bcmi.sjtu.edu.cn/~seed/index.html "")
*   [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html "")
*   [SEED-V](https://bcmi.sjtu.edu.cn/~seed/seed-v.html "")
*   [SEED-VII](https://bcmi.sjtu.edu.cn/home/seed/seed-vii.html "")
# File
*   Data process pipeline: utils.py
*   One-Hot processing: DataProcessing_OneHot.py
*   Gradients updating: StepwiseLR_GRL.py
*   MAT overall framework: MAT_framework.py
*   Module definition: Module_Definition.py
*   Pairwise learning: Pairwise_Learning_Module.py 
*   Model implementation: Implementation_MAT.py


