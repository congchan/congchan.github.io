---
title: The Evolution of Reward Modeling - From Human Feedback to Generative Inference-Time Scaling
date: 2025-05-25
mathjax: true
tags: ['2025', 'Large Language Model', 'Reward Modeling']
---

## **An Overview: The Critical Role of Reward Modeling in LLM Alignment**  

Reward modeling (RM) has emerged as a cornerstone of large language model (LLM) alignment, guiding models to align with human values and perform complex tasks. Early approaches relied heavily on Reinforcement Learning from Human Feedback (RLHF), but recent research has shifted toward more scalable, efficient, and generalizable RM frameworks. This blog explores the developmental arc of RM, connecting four seminal papers that have shaped the field: from Constitutional AI and self-evaluation mechanisms to inference-time scaling for generalist RM.  

### 1. Scoreing preference by parameters
Ouyang, Long, et al. Training Language Models to Follow Instructions with Human Feedback. arXiv:2203.02155, arXiv, 4 Mar. 2022. arXiv.org, http://arxiv.org/abs/2203.02155.

The paper presents a reward modeling (RM) approach as a core component of reinforcement learning from human feedback (RLHF) to align language models with human intent. Below is a detailed breakdown of the paper's views and methods on reward modeling:  

#### **1.1 Core Objectives of Reward Modeling**  
The reward model aims to:  
- **Quantify Human Preferences**: Convert subjective human judgments about model outputs into a scalar reward signal, enabling models to learn what constitutes "desirable behavior."  
- **Guide Model Alignment**: Direct language models to follow instructions, prioritize truthfulness, and avoid harmful outputs by optimizing against human-derived rewards.


#### **1.2. Data Collection for Reward Modeling**  
- **Input Source**: Prompts from the OpenAI API (filtered to remove PII) and labeler-written prompts, covering tasks like generation, QA, and summarization .  
- **Labeling Process**:  
  1. Labelers rank 4–9 model outputs per prompt from best to worst, generating pairwise comparisons (e.g., "Output A is preferred over Output B") .  
  2. To avoid bias, labelers undergo a screening test to assess sensitivity to sensitive content and alignment with research criteria .  


#### **1.3. Reward Model Architecture and Training**  
- **Model Structure**:  
  - Based on the GPT-3 architecture, initialized from a supervised fine-tuned (SFT) model with the final unembedding layer replaced by a projection layer to output a scalar reward .  
  - Uses a 6B parameter model for computational efficiency, as 175B models showed training instability .  

- **Training Methodology**:  
  - **Loss Function**: Cross-entropy loss to predict human-preferred outputs, formulated as:  
    $$
    \text{loss}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{\left(x, y_w, y_l\right) \sim D} \left[ \log \left( \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right) \right]
    $$
    where $y_w$\)$ and $y_l$ are the preferred and less preferred outputs, respectively, and $K$ is the number of outputs per prompt .  
  - **Batch Processing**: Treats all $\binom{K}{2}$ comparisons from a prompt as a single batch element to prevent overfitting and improve computational efficiency .  
  - **Normalization**: Adjusts rewards so that labeler demonstrations have a mean score of 0 before RL training .  


#### **1.4. Key Innovations and Insights**  
- **Generalization to Held-Out Labelers**: Reward models trained on one group of labelers generalize to new labelers, with cross-validation showing 69.6% accuracy in predicting preferences of unseen labelers .  
- **Trade-off with Public NLP Datasets**: RM-based RLHF may cause performance regressions on standard NLP tasks (e.g., SQuAD, DROP), but mixing pretraining gradients (PPO-ptx) mitigates this while preserving human preference .  
- **Role in InstructGPT**: The RM is crucial for improving model behavior: InstructGPT (PPO-ptx) outperforms GPT-3 despite having 100x fewer parameters, with 1.3B InstructGPT preferred over 175B GPT-3 in 85% of cases .  


#### **1.5. Limitations and Future Directions**  
- **Alignment Scope**: The RM aligns models to specific labelers and researchers, not broader human values, raising questions about fairness and representativeness .  
- **Toxicity and Bias**: While InstructGPT reduces toxicity, it shows minimal improvement on bias metrics (e.g., Winogender, CrowS-Pairs), indicating RM needs better signals for these dimensions .  
- **Scalability**: Future work may explore combining RM with adversarial data collection or constraint optimization to address harmful outputs and improve generalization .  

In summary, the paper demonstrates that reward modeling via RLHF is a powerful tool for aligning language models, but ongoing research is needed to address its limitations and expand its applicability to diverse human values.

### **2. Constitutional AI: Bootstrapping Harmlessness with AI Feedback**
Bai, Yuntao, et al. Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073, arXiv, 15 Dec. 2022. arXiv.org, https://doi.org/10.48550/arXiv.2212.08073.

In "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022), researchers introduced a paradigm that replaces human labels for harmfulness with AI-generated feedback. The approach uses a "constitution" of principles to guide self-critique and revision, enabling models to learn harmless behavior without direct human supervision.  

- **Key Innovation**: The framework combines supervised learning (critique → revision cycles) and RL from AI Feedback (RLAIF), where a preference model (PM) is trained on AI-generated comparisons. For example, models generate pairs of responses and evaluate which aligns better with constitutional principles (e.g., "avoid harmful advice").  
- **Impact**: As shown in Figure 2 of the paper, Constitutional AI achieves a Pareto improvement in harmlessness and helpfulness, outperforming RLHF models that trade off these traits. The approach reduces reliance on human labeling, a critical step toward scalable supervision.  

This work laid the groundwork for self-supervised RM, demonstrating that models can learn to evaluate their own behavior using explicit principles.  

### **3. DeepSeek-R1: Incentivizing Reasoning via Reinforcement Learning**
DeepSeek-AI, et al. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948, arXiv, 22 Jan. 2025. arXiv.org, https://doi.org/10.48550/arXiv.2501.12948.

The paper employs reward modeling strategically to enhance reasoning capabilities in LLMs through reinforcement learning (RL). Here’s a detailed breakdown of how reward modeling is utilized:

#### **3.1. Rule-Based Rewards for DeepSeek-R1-Zero**
DeepSeek-R1-Zero relies on a **rule-based reward system** to avoid the complexity and potential pitfalls of neural reward models. This system consists of two main components:
- **Accuracy Rewards**: Evaluate the correctness of responses. For example:
  - In math problems, the model must provide answers in a specified format (e.g., within a box) for rule-based verification.
  - In coding tasks (e.g., LeetCode), a compiler checks solutions against predefined test cases.
- **Format Rewards**: Enforce structural consistency by requiring the model to place reasoning processes between specific tags (e.g., `<think>`, `<\think>`, `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>` and `<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>`).

The paper explicitly avoids neural reward models (both outcome and process-based) for DeepSeek-R1-Zero, citing risks of **reward hacking** and the additional computational overhead of retraining reward models.

#### **3.2. Enhanced Reward Systems for DeepSeek-R1**
DeepSeek-R1 incorporates additional reward mechanisms to address readability and generalizability:
- **Language Consistency Reward**: Introduced to mitigate language mixing in Chain-of-Thought (CoT) reasoning. This reward measures the proportion of target language words in the CoT and is summed with accuracy rewards. While this slightly reduces reasoning performance, it improves human readability.
- **Multi-Stage RL with Diverse Rewards**:
  - **Reasoning-Oriented RL**: Uses rule-based rewards (accuracy + format) for tasks like math and coding.
  - **General Scenario RL**: Employs neural reward models to align with human preferences for helpfulness and harmlessness. For example:
    - **Helpfulness**: Focuses on the utility of the final summary.
    - **Harmlessness**: Evaluates the entire response (reasoning + summary) to prevent biased or harmful content.


#### **3.3. Reward Design for Cold Start and Distillation**
- **Cold Start Data**: Thousands of CoT examples are curated to fine-tune the model before RL. These examples include human-readable formats (e.g., summaries) and serve as a foundation for reward-aligned behavior.
- **Distillation**: The reasoning patterns of DeepSeek-R1 are distilled into smaller models using 800K training samples. While distillation itself does not use RL rewards, the teacher model (DeepSeek-R1) is trained with the aforementioned reward systems, ensuring smaller models inherit optimized reasoning behaviors.


#### **3.4. Key Trade-offs and Design Choices**
- **Rule-Based vs. Neural Rewards**: Rule-based rewards are prioritized for simplicity and to avoid reward hacking in large-scale RL. Neural rewards are introduced only when necessary (e.g., for general task alignment in DeepSeek-R1).
- **Balancing Performance and Readability**: The language consistency reward in DeepSeek-R1 trades off slight performance degradation for improved human interpretability, highlighting the importance of practical usability.


#### **3.5. Experimental Validation of Reward Models**
- **DeepSeek-R1-Zero**: Achieves significant reasoning gains (e.g., AIME 2024 Pass@1 from 15.6% to 71.0%) using purely rule-based rewards, demonstrating that complex reasoning can emerge without neural reward models.
- **DeepSeek-R1**: Outperforms DeepSeek-R1-Zero on readability and matches OpenAI-o1-1217 on reasoning tasks by combining rule-based and language consistency rewards.
- **Distilled Models**: Smaller models (e.g., 14B, 32B) trained on DeepSeek-R1’s rewarded outputs outperform state-of-the-art open-source models, validating the transferability of reward-aligned reasoning patterns.


#### **3.6 Conclusion**
The paper demonstrates that reward modeling in RL can be tailored to balance reasoning performance, readability, and human alignment. Rule-based rewards enable pure RL-driven reasoning emergence in DeepSeek-R1-Zero, while DeepSeek-R1 enhances this with language consistency and general preference rewards. This approach highlights the flexibility of reward systems in shaping LLM behavior without heavy reliance on neural reward models, paving the way for efficient and interpretable reasoning enhancements.

### **4. Inference-Time Scaling for Generalist Reward Modeling**
Liu, Zijun, et al. Inference-Time Scaling for Generalist Reward Modeling. arXiv:2504.02495, arXiv, 5 Apr. 2025. arXiv.org, https://doi.org/10.48550/arXiv.2504.02495.

Liu et al. (2025) address a critical gap in RM: generalizability across domains and efficient resource use. Their approach, Self-Principled Critique Tuning (SPCT), combines generative reward modeling (GRM) with inference-time scaling to improve RM performance without increasing training compute.  

- **Core Contributions**:  
  - **SPCT**: A two-phase RL method where models learn to generate adaptive principles and critiques, enhancing reward quality. For example, DeepSeek-GRM-27B with SPCT outperforms scalar RMs on benchmarks like Reward Bench and PPE (Table 2).  
  - **Inference-Time Scaling**: Parallel sampling and a meta RM guide voting on multiple reward samples, expanding the reward space and improving granularity. As shown in Figure 1, DeepSeek-GRM-27B with meta RM achieves 72.8% overall performance, surpassing models like Nemotron-4-340B-Reward.  
- **Connection to Prior Work**: SPCT builds on Constitutional AI’s self-critique and DeepSeek-R1’s RL efficiency, but shifts focus to generalist domains. The meta RM integrates self-evaluation insights (from Kadavath et al.) to filter low-quality rewards, aligning with P(IK)-like confidence metrics.  

### **The Developmental Arc: From Specialization to Generalization**  

The evolution of RM reflects a shift from human-dependent, task-specific approaches to self-supervised, generalizable frameworks:

1. **Early RLHF (2020s)**: Relied on massive human labeling, limited to specific domains.  
2. **Constitutional AI (2022)**: Introduced AI-generated feedback and principles, reducing human overhead.  
3. **Self-Evaluation (2022)**: Uncovered models’ ability to assess their own knowledge, enabling confidence-aware RM.
4. **Task-Oriented RL (2025, DeepSeek-R1)**: Optimized reasoning via RL, demonstrating task-specific RM scaling.  
5. **Generalist Inference-Time Scaling (2025, Liu et al.)**: Extended RM to diverse domains using generative models and inference-time compute, balancing efficiency and performance.  


### **Challenges and Future Directions**  
- **Bias and Calibration**: While SPCT reduces domain bias, models like DeepSeek-GRM still struggle with specific tasks (e.g., verifiable problems, Appendix B).  
- **Computational Overhead**: Inference-time scaling requires more compute per query, necessitating efficiency improvements.  
- **Cross-Domain Generalization**: Combining task-specific RM (DeepSeek-R1) with generalist GRM (Liu et al.) remains an open challenge.  

Future work may integrate self-supervised RM with external tools (e.g., code execution for verification) and explore hybrid frameworks that balance training and inference-time scaling.  


### **Conclusion**  

The landscape of reward modeling is evolving rapidly, driven by innovations that prioritize scalability, self-supervision, and generalizability. From Constitutional AI’s principles to SPCT’s inference-time scaling, these methods collectively push LLMs toward more aligned, transparent, and efficient behavior. As shown in the comparative results across papers, the field is moving toward a future where RM serves as a versatile interface for LLM alignment, enabling models to reason, evaluate, and adapt to diverse human needs.  

  
### **Key Citations**
- Ouyang, Long, et al. Training Language Models to Follow Instructions with Human Feedback. arXiv:2203.02155, arXiv, 4 Mar. 2022. arXiv.org, http://arxiv.org/abs/2203.02155.
- Bai, Yuntao, et al. Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073, arXiv, 15 Dec. 2022. arXiv.org, https://doi.org/10.48550/arXiv.2212.08073.
- DeepSeek-AI, et al. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948, arXiv, 22 Jan. 2025. arXiv.org, https://doi.org/10.48550/arXiv.2501.12948.
- Liu, Zijun, et al. Inference-Time Scaling for Generalist Reward Modeling. arXiv:2504.02495, arXiv, 5 Apr. 2025. arXiv.org, https://doi.org/10.48550/arXiv.2504.02495.
- Kadavath, Saurav, et al. Language Models (Mostly) Know What They Know. arXiv:2207.05221, arXiv, 21 Nov. 2022. arXiv.org, http://arxiv.org/abs/2207.05221.
