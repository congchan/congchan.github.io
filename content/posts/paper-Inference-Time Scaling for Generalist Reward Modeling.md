---
title: Paper Reading - Inference-Time Scaling for Generalist Reward Modeling
date: 2025-05-05
mathjax: true
tags: ['Readings', '2025', 'Large Language Model', 'Reward Modeling']
---
Liu, Zijun, et al. Inference-Time Scaling for Generalist Reward Modeling. arXiv:2504.02495, arXiv, 5 Apr. 2025. arXiv.org, https://doi.org/10.48550/arXiv.2504.02495.

#### Problem Statement
Reinforcement Learning (RL) has become pivotal in post-training large language models (LLMs), but generating accurate reward signals for diverse domains remains challenging. Existing reward models (RMs) often rely on human-designed rules or verifiable tasks, struggling with generalizability and inference-time scalability. This paper addresses how to improve RM effectiveness through increased inference compute and adaptive learning methods for general queries.

#### Key Contributions
1. **Self-Principled Critique Tuning (SPCT)**: A novel learning method to foster scalable reward generation in generative RMs (GRMs) by enabling adaptive principle and critique generation via rule-based online RL.
2. **Pointwise Generative Reward Modeling (GRM)**: Adopted for flexibility in handling various input types and potential for inference-time scaling through diverse reward generation.
3. **Inference-Time Scaling Strategies**: Parallel sampling to expand compute usage and a meta RM to guide voting, enhancing scaling performance without severe biases.

#### Methodology
- **SPCT Framework**:
  - **Rejective Fine-Tuning (RFT)**: Cold start to align GRM with correct format and input types, rejecting incorrect or trivial trajectories.
  - **Rule-Based RL**: Optimizes principle and critique generation using GRPO, encouraging the model to distinguish best responses via online-optimized criteria.
- **Inference-Time Scaling**:
  - **Parallel Sampling**: Generates multiple principle-critique pairs, voting on final rewards to expand the value space.
  - **Meta RM**: A scalar RM trained to filter low-quality samples, guiding voting for more accurate results.

#### Experimental Results
- **Performance Benchmarks**: DeepSeek-GRM outperforms scalar/semi-scalar RMs (e.g., Nemotron-4-340B-Reward, GPT-4o) on Reward Bench, PPE, and RMB without domain biases.
- **Inference-Time Scalability**: Voting with 32 samples and meta RM guidance achieves up to 72.8% overall accuracy, surpassing training-time scaling (e.g., 671B model performance with 27B model + inference scaling).
- **Ablation Studies**: Principle generation and non-hinted sampling prove critical; meta RM effectively filters low-quality outputs.

#### Conclusion
SPCT enhances GRMs to generate adaptive, high-quality rewards, demonstrating that inference-time scaling can outperform model size scaling. Future work will focus on integrating tools, improving efficiency, and offline evaluation applications. The models will be open-sourced to advance generalist reward systems.

#### Key Figures/Tables
- **Figure 1**: Inference-time scaling performance across benchmarks, showing DeepSeek-GRM’s superiority with increased samples.
- **Table 2**: Overall results comparing DeepSeek-GRM against public models and baselines, highlighting its competitive edge.
- **Table 4**: Ablation study verifying the importance of SPCT components (e.g., principle generation, rejective sampling).

This work bridges the gap between RM generalizability and compute efficiency, offering a scalable path for LLM alignment in diverse domains.