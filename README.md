<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> TimeMAE: Self-supervised Representation of Time Series with Decoupled Masked Autoencoders </b></h2>
</div>

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```
@article{cheng2023timemae,
  title={Timemae: Self-supervised representations of time series with decoupled masked autoencoders},
  author={Cheng, Mingyue and Liu, Qi and Liu, Zhiding and Zhang, Hao and Zhang, Rujiao and Chen, Enhong},
  journal={arXiv preprint arXiv:2303.00320},
  year={2023}
}
```

## Updates/News:

## Introduction

In this work, we propose TimeMAE, a novel self-supervised paradigm for learning transferrable time series representations based on transformer networks. The distinct characteristics of the TimeMAE lie in processing each time series into a sequence of non-overlapping sub-series via window-slicing partitioning, followed by random masking strategies over the semantic units of localized sub-series. Such a simple yet effective setting can help us achieve the goal of killing three birds with one stone, i.e., (1) learning enriched contextual representations of time series with a bidirectional encoding scheme; (2) increasing the information density of basic semantic units; (3) efficiently encoding representations of time series using transformer networks. Nevertheless, it is a non-trivial to perform reconstructing task over such a novel formulated modeling paradigm. To solve the discrepancy issue incurred by newly injected masked embeddings, we design a decoupled autoencoder architecture, which learns the representations of visible (unmasked) positions and masked ones with two different encoder modules, respectively. Furthermore, we construct two types of informative targets to accomplish the corresponding pretext tasks. One is to create a tokenizer module that assigns a codeword to each masked region, allowing the masked codeword classification (MCC) task to be completed effectively. Another one is to adopt a siamese network structure to generate target representations for each masked input unit, aiming at performing the masked representation regression (MRR) optimization. Comprehensively pre-trained, our model can efficiently learn transferrable time series representations, thus benefiting the classification of time series


<img width="1000" alt="截屏2024-09-13 11 17 43" src="https://github.com/user-attachments/assets/e5cef2f4-b2b8-4a53-afa3-845b1f6d3d0d">

<img width="1000" alt="截屏2024-09-13 11 18 07" src="https://github.com/user-attachments/assets/a52df37c-6535-48a3-9b26-bc403e5233d5">

<img width="1000" alt="截屏2024-09-13 11 20 12" src="https://github.com/user-attachments/assets/6eec9815-036f-4393-a3f8-bd5ca56f3309">




