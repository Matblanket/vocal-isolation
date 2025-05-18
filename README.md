# 🎧 Vocal Source Separation

This project focuses on isolating vocals from instrumental components in music using deep learning techniques.

## 📄 Project Report

All details, including methodology, experiments, and results, are documented in the [report.ipynb](./report.ipynb) notebook.

## 📁 Directory Structure

```
.
├── report.ipynb            # Main project report with code and explanations
├── ProjectFiles/           # Contains all experiment code and configurations
└── README.md               # This file
```

## 📚 Background

This project was completed as part of the Conversational AI course taught by Mirco Ravanelli. The course provided a strong foundation in state-of-the-art methods for speech and audio processing, with a particular emphasis on deep learning techniques. Key topics included:

Transformer architectures for audio modeling

Advanced models such as DeepSeek, Mixture of Experts (MoE), and RoPE (Rotary Position Embedding)

Hands-on experience with SpeechBrain, an open-source conversational AI toolkit

The knowledge and tools gained from this course directly informed the approach to vocal source separation used in this project.

## 💻 Compute Environment

All model training and experimentation were performed on Paperspace by DigitalOcean, leveraging GPU-backed virtual machines. Depending on availability, the experiments used a mix of:

- NVIDIA A5000

- NVIDIA P5000

This setup enabled efficient training of deep learning models on high-resolution audio data, supporting faster iteration and more robust experimentation.

### 📦 Dataset

The primary dataset used is **MUSDB18**, a standard benchmark dataset for music source separation that contains professionally mixed audio tracks with isolated vocal and instrumental stems.

> **Citation**
> Rafii, Zafar, et al. *The MUSDB18 corpus for music separation*. 2017.
> DOI: [10.5281/zenodo.1117372](https://doi.org/10.5281/zenodo.1117372)

### 📘 Related Work and Tools

This work draws on the capabilities and research foundations of **SpeechBrain**:

> Ravanelli, Mirco, et al. *Open-Source Conversational AI with SpeechBrain 1.0*. Journal of Machine Learning Research, 2024, Vol. 25, No. 333.
> [http://jmlr.org/papers/v25/24-0991.html](http://jmlr.org/papers/v25/24-0991.html)

> Ravanelli, Mirco, et al. *SpeechBrain: A General-Purpose Speech Toolkit*. arXiv preprint arXiv:2106.04624, 2021.
> [https://arxiv.org/abs/2106.04624](https://arxiv.org/abs/2106.04624)
