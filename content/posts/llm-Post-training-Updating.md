---
title: An overview of Post-training algorithms for large language model (LLM)
date: 2025-05-30
tags: ['2025', 'Large Language Model', 'Post-training']
---

In the race to build truly helpful AI assistants, we've discovered a fundamental truth: raw intelligence isn't enough. A model that masters calculus but can't refuse harmful requests is like a library with no librarian - overflowing with knowledge but dangerously uncurated.  

This is the alignment problem: how do we transform raw language models into trustworthy collaborators? For years, **Reinforcement Learning from Human Feedback (RLHF)** reigned supreme. Its PPO-based approach taught ChatGPT to decline malicious requests and helped Claude write harmless poetry. But beneath the surface, RLHF's complexity was showing:  

1. **The 3-stage training treadmill** (SFT → Reward Modeling → RL tuning)  
2. **Prohibitively expensive** human preference labeling  
3. **Reward hacking** vulnerabilities where models "game" the system  

Enter the new generation of alignment techniques. There are three trends of directions:  
1. Eliminating reward modeling stages. Or use Rule-based rewards to incentivize LLM intelligence.
2. Using AI-generated preferences.
3. Enabling single-step optimization  

I am currently following the most cutting-edge LLM alignment methods, and this blog will be updated periodically.


## **I. RL\*F (Reinforcement Learning from X Feedback) with Proximal Policy Optimization (PPO)**
Proximal Policy Optimization (PPO) is the mose widely used reinforcment learning algorithm in Post-training. PPO is a policy gradient algorithm that optimizes policies by maximizing a clipped surrogate objective to balance exploration and exploitation. It is widely used in RLHF due to its stability and sample efficiency.  
- **Core Innovation**: Uses clipped objective function to limit policy updates, balancing stability and performance. Dominates RLHF pipelines.  
- **Application**: OpenAI's ChatGPT, DeepSeek-R1, Claude series .  
- **Limitations**: Requires separate reward model training; unstable with large batches.  

**Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).

### **Reinforcement Learning from Human Feedback (RLHF)**
RLHF is a machine learning technique that combines reinforcement learning with human preferences to train AI models, particularly large language models (LLMs), to produce outputs that are more aligned with human values and expectations.

RLHF combines supervised fine-tuning (SFT) with PPO, using human preferences to train a reward model (RM) that guides policy updates. RLHF aligns LLMs with human values but is costly due to manual labeling.  

**Paper**: [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155).  

#### Key Components and Workflow
![](/images/RLHF.png)

1. **Base Model Training**
    - Start with a pre-trained language model (like GPT)
    - The model is initially trained on large datasets using supervised learning
1. **SFT**: Supervised fine-tuning on high-quality data.
1. **Human Feedback Collection**
    - Human evaluators compare pairs of model outputs
    - They rank which response is better based on criteria like: Helpfulness,Harmlessness, Honesty
1. **Reward Modeling**:
    - A separate neural network (reward model) is trained on human preference rankings, to predict human preferences
    - This model learns to score outputs based on the collected human feedback
    - It essentially learns to mimic human judgment
1. **Reinforcement Learning Optimization**
    - The original language model as policy is optimized against RM using PPO.
    - The reward model provides feedback signals
    - Techniques like Proximal Policy Optimization (PPO) are commonly used
    - The model learns to generate responses that maximize the predicted human preference score

#### Benefits
- Better Alignment: Models produce outputs more consistent with human values
- Reduced Harmful Content: Helps minimize toxic, biased, or dangerous responses
- Improved Quality: Responses become more helpful and relevant
- Scalability: Once trained, the reward model can provide feedback without constant human intervention

#### Challenges
- Scalability of Human Feedback: Collecting sufficient high-quality human feedback is expensive and time-consuming
- Reward Hacking: Models might find ways to maximize reward scores without actually improving quality
- Bias in Human Feedback: Human evaluators may introduce their own biases
- Complexity: The multi-stage training process is computationally intensive

### **Reinforcement Learning from AI Feedback (RLAIF)**
In **Bai, Yuntao, et al. Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073, arXiv, 15 Dec. 2022. arXiv.org, https://doi.org/10.48550/arXiv.2212.08073.**, researchers introduced a paradigm that replaces human labels for harmfulness with AI-generated feedback. The approach uses a "constitution" of principles to guide self-critique and revision, enabling models to learn harmless behavior on a hybrid of human and AI preferences. This paper was the first effort to explore RLAIF.

- **Key Innovation**: The framework combines supervised learning (critique → revision cycles) and RL from AI Feedback (RLAIF), where a preference model (PM) is trained on AI-generated comparisons. For example, models generate pairs of responses and evaluate which aligns better with constitutional principles (e.g., "avoid harmful advice").  
- **Impact**: As shown in Figure 2 of the paper, Constitutional AI achieves a Pareto improvement in harmlessness and helpfulness, outperforming RLHF models that trade off these traits. The approach reduces reliance on human labeling, a critical step toward scalable supervision.  

This work laid the groundwork for self-supervised RM, demonstrating that models can learn to evaluate their own behavior using explicit principles.  

In [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267), RLAIF achieved comparable performance to RLHF.

## **II. Improve Value Functioning and Eliminating Critic Model**
The PPO algorithm necessitates loading four models, each of substantial size, which introduces considerable engineering complexity in the design of multi-model training, inference, and real-time parameter updates. This process demands a significant amount of GPU resources.

For instance, during RLHF training, when the Actor, Critic, Reward, and Ref Models are of identical scale, such as 70B, employing vLLM/TensorRT-llm for PPO sample generation acceleration and DeepSpeed/Megatron for training acceleration results in roughly equal computational resource consumption between inference and training stages. Consequently, the Critic model accounts for approximately one-quarter of the total computational resource usage.

In the context of LLMs, it is common for only the final token to receive a reward score from the reward model. This practice can complicate the training of a value function that accurately reflects each token's contribution. To address this challenge, numerous studies focus on optimizing the calculation of the value function, incidentally simplifying or potentially eliminating the need for the Critic model in the process.

### **Group Relative Policy Optimization (GRPO)**  
Group Relative Policy Optimization (GRPO) is an efficient reinforcement learning algorithm proposed in the DeepSeekMath paper to enhance mathematical reasoning in language models.

**Paper**: *Shao, Zhihong, et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, arXiv, 27 Apr. 2024. arXiv.org, http://arxiv.org/abs/2402.03300.*.  


![](/images/GRPO.png)

GRPO is a variant of Proximal Policy Optimization (PPO) that eliminates the need for a critic model (value function), instead estimating the baseline from **group-averaged rewards**. This reduces memory and computational costs significantly, making it more resource-efficient.

**Key Differences from PPO**
- **No Value Function**: Unlike PPO, which uses a learned value function to compute advantages, GRPO calculates advantages using relative rewards within a group of sampled outputs for the same question.
- **Group-based Baseline**: For each question $q$, GRPO samples $ G $ outputs from the old policy. The baseline is the average reward of these outputs, and advantages are normalized within the group. 
- **Simplified Objective**: GRPO optimizes the policy by maximizing a objective that uses group-relative advantages, avoiding the complexity of value function training.

#### **How GRPO Works**
1. **Sampling**: For each question $q$, sample $G$ outputs $\{o_1, o_2, \dots, o_G\}$ from the old policy $\pi_{\theta_{\text{old}}}$.
1. **Outcome Reward Scoring and Normalization**: Use a reward model to score each output, yielding $\{r_1, r_2, \dots, r_G\}$. Outcome supervision provides the normalized reward at the end of each output $o_i$ and sets the advantages $\hat{A}_{i,t}$ of all tokens in the output as the normalized reward by subtracting the group mean and dividing by the standard deviation:  
   
   $$
    \hat{A}_{i,t} = \tilde{r}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}
   $$

1. **Policy Update**: Maximize the GRPO objective, which includes a KL divergence term to regularize against a reference model. optimizes the policy model by maximizing the following objective:
![](/images/GRPO-obj.png)

    $$
    \begin{aligned} \mathcal{J}_{\text{GRPO}}(\theta) &= \mathbb{E}\left[ q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q) \right] \\\\ 
        & \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left[ \frac{\pi_{\theta}(o_{i,t}|q, o_{i,\lt t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,\lt t})} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_{\theta}(o_{i,t}|q, o_{i,\lt t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,\lt t})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}_{i,t} \right] - \beta \mathbb{D}_{\text{KL}} \left[ \pi_{\theta} \| \pi_{\text{ref}} \right] \right\} \end{aligned}
    $$

    where $\epsilon$ and $\beta$ are hyper-parameters, and $\hat{A}_{i,t}$ is the advantage calculated based on relative rewards of the outputs inside each group only, which will be detailed in the following subsections. 
    - Also note that, instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the loss, avoiding complicating the calculation of $\hat{A}_{i,t}$.
    - And different from the KL penalty term used in PPO, GRPO estimate the KL divergence with the following unbiased estimator (J. Schulman. Approximating kl divergence, 2020. URL http://joschu.net/blog/kl-approx.html.):

    $$
    \mathbb{D}_{KL} \left[ \pi_{\theta} \| \pi_{ref} \right] = \frac{\pi_{ref}(o_{i,t}|q, o_{i,\lt t})}{\pi_{\theta}(o_{i,t}|q, o_{i,\lt t})} - \log \frac{\pi_{ref}(o_{i,t}|q, o_{i,\lt t})}{\pi_{\theta}(o_{i,t}|q, o_{i,\lt t})} - 1
    $$
    which is guaranteed to be positive.


**Experimental Results**
- DeepSeekMath-RL (7B) using GRPO surpasses all open-source models on MATH and approaches closed-source models like GPT-4.
- Iterative GRPO (updating the reward model incrementally) further boosts performance, especially in the first iteration.
- GRPO with process supervision (step-wise rewards) outperforms outcome supervision, highlighting the value of fine-grained feedback.

#### **Combining GRPO and RLHF - Deepseek-R1**
DeepSeek-R1 is the first public model who make use of both rule-based GRPO RL and general RLHF alignment.

**DeepSeek-R1-Zero: Pure RL Training**
- **RL Algorithm**: Uses Group Relative Policy Optimization (GRPO) to optimize policies without a critic model, reducing training costs.
- **Reward Modeling**: Relies on rule-based rewards for accuracy (e.g., math problem correctness) and format (enforcing CoT within tags), avoiding neural reward models to prevent reward hacking.
- **Self-Evolution**: Through RL, the model autonomously develops complex reasoning behaviors, such as reflecting on mistakes and exploring alternative solutions, leading to significant performance gains. For example, AIME 2024 pass@1 improves from 15.6% to 71.0%, and to 86.7% with majority voting.
- **Limitation**: 
  - Suffering from readability issues, such as language mixing and unstructured outputs. And
  - Lack of non-reasoning tasks such as writing and factual QA is suboptimal. 
  - Experiences an unstable early training process. 

**DeepSeek-R1: Cold-Start and Multi-Stage Refinement**
DeepSeek-R1 is an attempts to address the limitations of DeepSeek-R1-Zero. 
- **1. Cold-Start Data**: Collect thousands of long CoT (chain-of-thought) examples through few-shot prompting, model-generated outputs, and human annotation. These cold-start data are used to fine-tune the DeepSeek-V3-Base to create an initial RL actor that prioritizes readable formats (e.g., summaries and structured CoT) and reducing language mixing.

- **2. Reasoning-Oriented RL**: Apply the same large-scale reinforcement learning training process as employed in DeepSeek-R1-Zero. Incorporates language consistency rewards to mitigate mixed-language outputs, balancing performance with human readability.

- **3. Rejection Sampling & SFT**: After RL convergence, new SFT data is collected from RL checkpoints, combining reasoning and non-reasoning tasks (e.g., writing, factual QA) to enhance general capabilities.
  - For **Reasoning Data Collection**: Use rejection sampling on the RL checkpoint to curate ~600K reasoning samples, filtering out mixed-language and unreadable CoT. Include generative reward models (using DeepSeek-V3) for evaluation.
  - For **Non-Reasoning Data**: Reuse SFT data from DeepSeek-V3 for tasks like writing, factual QA, and translation, collecting ~200K samples.
  - **Fine-Tuning**: Train DeepSeek-V3-Base on the combined ~800K samples for two epochs to enhance general capabilities.

- **4. Scenario-Agnostic RL**: A final RL stage aligns the model with human preferences for helpfulness and harmlessness, using a mix of rule-based and neural reward models.
  - **Reasoning Data**: Use rule-based rewards for math, code, and logic tasks, as in previous stages.
  - **General Data**: Employ reward models to capture human preferences in complex scenarios (e.g., writing, role-playing), building on DeepSeek-V3’s pipeline .
  - **Evaluation Focus**: For helpfulness, assess the final summary; for harmlessness, evaluate the entire response (CoT and summary) to mitigate risks .

**Key Advantages of Iterative RL Training**
- **Performance Enhancement**: DeepSeek-R1 achieves comparable results to OpenAI-o1-1217 on reasoning benchmarks (e.g., 79.8% pass@1 on AIME 2024) .
- **Readability and Consistency**: Cold-start data and language rewards reduce language mixing and improve output structure .
- **Generalization**: SFT with diverse data enables competence in non-reasoning tasks like creative writing and factual QA .

**Performance Benchmarks**
DeepSeek-R1 and distilled models excel on reasoning tasks:
- **Math/Coding**: AIME 2024 pass@1 of 79.8% (vs. OpenAI-o1-1217’s 79.2%), MATH-500 pass@1 of 97.3%, and Codeforces rating of 2029 (top 3.7% of human participants).
- **Knowledge Tasks**: MMLU score of 90.8%, GPQA Diamond of 71.5%, slightly below o1-1217 but surpassing other closed-source models.
- **General Tasks**: Strong performance in creative writing, summarization, and long-context understanding, with win rates of 87.6% on AlpacaEval 2.0 and 92.3% on ArenaHard.

### **RLOO (REINFORCE Leave One-Out)**
PPO suffers from high computational costs and sensitive hyperparameter tuning. This paper argues that a simpler RL methods, specifically REINFORCE-style optimization, can preserve or even enhance performance while reducing complexity.

**Paper**: Ahmadian, Arash, et al. Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs. arXiv:2402.14740, arXiv, 26 Feb. 2024. arXiv.org, https://doi.org/10.48550/arXiv.2402.14740.

![Image from https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo](/images/RLOO.png)
![Image from https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo](/images/RLOO-1.png)

Key Insights:
1. PPO's Limitations in RLHF: PPO was designed for traditional RL environments with high variance and random policy initializations. In RLHF, pre-trained LLMs provide a strong policy initialization, concentrating probability mass on a small subset of tokens. This stability makes PPO's complexity (e.g., clipping, value networks) unnecessary.
2. REINFORCE for RLHF: By modeling the entire sequence generation as a single action (vs. PPO's token-level actions), REINFORCE directly optimizes the full trajectory reward with unbiased baselines. This avoids the bias introduced by PPO's bootstrapped value functions.
3. REINFORCE Leave-One-Out (RLOO): A multi-sample extension of REINFORCE, RLOO uses online samples to create dynamic baselines, reducing variance without bias. It outperforms PPO and RL-free methods by fully leveraging all generated samples.

#### REINFORCE
REINFORCE loss, which applies the vanilla policy gradient to the entire sequence, using a moving average reward as a baseline.It basically multiplies the (reward - baseline) by the logprob of actions.
- Core Idea: In LLM applications, since the reward $r(x, y)$ is only available at the end of the full sequence, REINFORCE models the entire generation as a single action rather than each token. This aligns with the bandit problem formulation, where the Markov Decision Process (MDP) includes only the initial state (prompt) and the terminal state (completed sequence).
- Estimator: It uses the REINFORCE estimator to backpropagate through the discrete action space (generation) and directly optimize the KL-shaped reward objective for the entire sequence. The update rule is  
    $$
    \begin{equation}
        \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(. | x)}\left[R(y, x) \nabla_{\theta} \log \pi_{\theta}(y | x)\right]
        \tag{6}
    \end{equation}
    $$
- Baseline: To improve learning, one can reduce the variance of the REINFORCE estimator, while keeping it unbiased, by subtracting a baseline $b$ that has high covariance with the stochastic gradient estimate of the Eq.6:
    $$
    \begin{equation}
        \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(.|x)} \big[ (R(y, x) - b) \nabla_\theta \log \pi_\theta(y|x) \big]
        \tag{7}
    \end{equation}
    $$
    The moving average of all rewards throughout training (Williams, 1992) is a strong parameter-free choice for the baseline:
    $$
    \begin{equation}
        b_{\text{MA}} = \frac{1}{S} \sum_{s} R(x^s, y^s)
        \tag{8}
    \end{equation}
    $$
    Where $S$ is the number of training steps, and $(x^s, y^s)$ is the prompt-completion pair at the step $s$. This baseline is simple, computationally cheap, and parameter-free.

Noticed that REINFORCE is a special case of PPO (Huang, Shengyi, et al. A2C Is a Special Case of PPO. arXiv:2205.09123, arXiv, 18 May 2022. arXiv.org, https://doi.org/10.48550/arXiv.2205.09123.)

Even though the logprob is explicitly in the REINFORCE loss, it is also implicitly in the PPO loss.


#### REINFORCE Leave-One-Out (RLOO)
Leverages multiple online samples to further reduce variance in the REINFORCE estimator while keeping it unbiased. 

For each prompt, generates k samples and uses the average reward of k-1 samples as a baseline for the remaining one, creating a variance-reduced gradient estimate.

The baseline in Eq. 8 is simple to implement and computationally cheap. However, it can be improved upon if we have access to multiple online samples, that can be used for further unbiased variance reduction: 
1. The rewards for each sample can serve all other samples as a baseline. 
2. Policy updates can be done on an average of gradient estimates for each sample, resulting in a variance-reduced multi-sample Monte-Carlo (MC) estimate. 

This is the intuition behind the REINFORCE Leave-One-Out (RLOO) estimator, proposed by (Kool et al., 2019):
$$
    \frac{1}{k} \sum_{i=1}^{k} \left[ R(y_{(i)}, x) - \frac{1}{k - 1} \sum_{ j \neq i} R(y_{(j)}, x) \right] \nabla \log \pi(y_{(i)} | x) \text{ for } y_{(1)}, \ldots, y_{(k)} \stackrel{i.i.d}{\sim} \pi_{\theta}(. | x)
$$
Where \(k\) refers to the number of online samples generated, \(\text{RLOO}_k\) considers each \( y_{(i)} \) individually and uses the remaining \( k - 1 \) samples to create an unbiased estimate of the expected return for the prompt, akin to a parameter-free value-function, but estimated at each training step. 

This is a much more effective baseline (as the paper's experiments showed) than \( b_{\text{MA}} \) since it's created on-the-fly for each sample and at each training step, but comes at a cost of increased sampling time during training. 

Noted that generating extra samples as a means of variance reduction has been proposed by concurrent work (Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models, 2023.), but RLOO focus on the efficiency benefits of fully utilizing all samples.

#### Results
1. **Performance**: REINFORCE outperforms PPO by 3.2–20.3% in win-rate. RLOO further improves performance, surpassing DPO and RAFT across all datasets.
2. **Sample Efficiency**: RLOO with k=2 matches or exceeds RAFT with k=4, demonstrating better use of online samples.
3. **Robustness**: RLOO is less sensitive to KL penalty and reward noise compared to RAFT, maintaining stable performance under varying conditions.
4. **Alignment Tax**: RLOO preserves language fluency (perplexity) and diversity better than PPO, with lower reward variance—a key factor for safety-critical applications.

### REINFORCE++
RLOO and GRPO increase inference costs to trade for eliminating the critic model. The REINFORCE++ authors argue that eliminating the critic model may inadvertently lower training efficiency due to increased inference costs.

**Paper**: Hu, Jian, et al. REINFORCE++: An Efficient RLHF Algorithm with Robustness to Both Prompt and Reward Models. arXiv:2501.03262, arXiv, 6 Apr. 2025. arXiv.org, https://doi.org/10.48550/arXiv.2501.03262.

The blog https://hijkzzz.notion.site/unraveling-rlhf-and-its-variants-engineering-insights argues that:
- When all models (Actor, Critic, Reward, Reference) are similar in scale (e.g., 70B parameters), inference and training consume roughly equal computational resources (1:1 ratio)
- Eliminating the critic model may actually **reduce training efficiency** due to increased inference costs
- System complexity remains largely unchanged since multiple models still need to operate together
- **Performance Analysis:**
    -  REINFORCE-based methods (e.g., RLOO, ReMax, GRPO) eliminate the critic but struggle with accurate advantage estimation, often overfitting to simple prompts and being vulnerable to reward hacking. These methods estimate advantages per prompt, leading to instability and poor generalization.  
    - GRPO and RLOO don't provide significant theoretical improvements over PPO
    - The claimed advantages (like "10x" efficiency gains) are often exaggerated
    - PPO can address critic model issues by initializing critics with actor weights
    - **Alternative solution**: Pre-train the critic model by freezing actor weights during PPO training
- **Technical Issues with GRPO:**
    - **Numerical instability**: Small differences in sampled rewards can be amplified during normalization
    - Example: rewards of 1.001 vs 1.00 become -0.7070 vs 0.7072 after normalization
    - **Convergence problems**: When all sampled rewards are equal, GRPO provides zero learning signal
- Under Process Reward Model (PRM) settings, GRPO essentially becomes REINFORCE with mean baseline
- **Bottom Line:**: Both GRPO and RLOO are most beneficial when critic/reward models are significantly larger than actors, but even then, PPO remains a viable alternative with proper initialization strategies. The computational and complexity advantages are often overstated.

REINFORCE++ is a critic-free RLHF algorithm that uses the **global batch mean reward as a baseline** instead of prompt-specific baselines, preventing overfitting and improving robustness.


![](/images/REINFORCE++.png)
The overall algorithm flow of REINFORCE++: Sample one response per prompt, compute rewards, normalize advantages, and update the policy using a clipped objective (similar to PPO but without the critic).  
- Advantages Normalization: Normalizes advantages across the entire batch to stabilize training and enhance out-of-distribution (OOD) generalization. REINFORCE++ replaces prompt-specific baselines with the mean reward of a global batch, reducing overfitting to individual prompts. The Advantage is calculated as:
    $$
    A_{q, o_t} = r(o_{1:t}, q) - \beta \cdot \sum_{i=t}^T KL(i), \quad \text{with } KL(t) = \log\left(\frac{\pi_{\theta_{\text{old}}}^{RL}(o_t | q, o_{\lt t})}{\pi^{\text{SFT}}(o_t | q, o_{\lt t})}\right)
    $$
    The token-level KL penalty avoids the need for a critic network while achieving comparable stability. The gradient of the token-level KL penalty has been theoretically proven to be unbiased concerning the $k_3$ loss of GRPO in RLHF. 

    The advantage is normalized globally:
    $$
    A_{q, o_t}^{\text{norm}} = \frac{A_{q, o_t} - \text{mean}(A_{q, o_t})}{\text{std}(A_{q, o_t})}
    $$


#### Experimental Results
- Bradley-Terry Reward Model: REINFORCE++ matches or exceeds performance of GRPO, RLOO, and ReMax on OOD benchmarks (e.g., GSM8K, MATH, code generation), with higher per-token efficiency.  
- Long CoT Tasks:  
  - Small-Scale Datasets: GRPO overfits to training prompts (e.g., AIME-24), achieving near-perfect scores but failing on OOD test sets (AIME-25), while REINFORCE++ shows stable generalization.  
  - Logical Reasoning (Knights and Knaves): REINFORCE++ outperforms GRPO in complex and OOD scenarios (e.g., 8-character puzzles), with higher Pass@1 scores (36 vs. 20) and longer, more reasoned responses.  
  - Mathematical Reasoning: From scratch or fine-tuned models, REINFORCE++ demonstrates better OOD generalization on MATH and AIME datasets.  

### ReMax
PPO introduces significant computational overhead for LLMs due to its complex architecture: it requires training a value model, leading to heavy memory usage, tedious hyperparameter tuning, and slow training. To make RLHF efficient, ReMax leverages 3 properties of RLHF: fast simulation, deterministic transitions, and trajectory-level rewards. ReMax does not require training an additional value model as in PPO and is further enhanced with a new variance reduction technique.

**Paper**: Li, Ziniu, et al. ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models. arXiv:2310.10505, arXiv, 16 May 2024. arXiv.org, https://doi.org/10.48550/arXiv.2310.10505.

Key limitations of PPO for RLHF:  
- Value model overhead: Consumes ~46% of GPU memory for a 7B model.  
- Hyperparameter complexity: Requires tuning 4+ parameters (e.g., clipping, GAE coefficient).  
- Slow convergence: Training with PPO can be 4× slower than earlier RLHF steps.  

![](/images/ReMax.png)

#### Key Insights: Unique Properties of RLHF for LLMs
ReMax leverages three properties of RLHF that PPO overlooks:  
1. Fast simulation: Generating a complete LLM response (trajectory) is rapid (e.g., <10s for 7B models).  
2. Deterministic transitions: Text generation depends only on past tokens, with no stochastic environment dynamics.  
3. Trajectory-level rewards: Rewards are given only after the full response, not at each step.  

#### **The ReMax Algorithm**  
ReMax is built on the REINFORCE algorithm but introduces a variance reduction technique:  
- Greedy baseline: For each prompt, compute the reward of a greedy (deterministic) response and use it to normalize the gradient, reducing variance.
    1. For a prompt, sample a stochastic response and a greedy response.  
    2. Compute the reward difference between the two responses.  
- No value model: Directly optimizes the policy to maximize the log-likelihood of high-reward responses, weighted by the reward difference.   

Pseudo-code Core Steps:
```
for prompt in dataset:
    seq = lm.sample(prompt, greedy=False)          # Stochastic response
    seq_max = lm.sample(prompt, greedy=True)      # Greedy response
    rew = rm(prompt, seq) - rm(prompt, seq_max)   # Reward difference
    logp = lm.inference(prompt, seq)              # Log-likelihood
    loss = - (logp.sum() * rew).mean()            # Loss for optimization
    lm.minimize(loss)
```  

### **DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)**  
DAPO decouples policy clipping and dynamic sampling to enhance training efficiency, reducing variance in gradient estimates.

**Paper**: Yu, Qiying, et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476, arXiv, 20 May 2025. arXiv.org, https://doi.org/10.48550/arXiv.2503.14476.

DAPO addresses critical challenges in RL training—such as entropy collapse, reward noise, and instability—with four key techniques.
1. Clip-Higher: Decouples lower $\varepsilon_{low}$ and higher $\varepsilon_{high}$ clipping ranges in policy optimization to prevent entropy collapse. By increasing $\varepsilon_{high}$, low-probability "exploration" tokens gain more room for probability increases, enhancing diversity. <!--  (Figure 2). --> ![](/images/DAPO-1.png)
1. Dynamic Sampling: Over-samples and filters out prompts with all-correct or all-wrong outputs to maintain effective gradient signals. This mitigates gradient-decreasing issues from zero-advantage batches, improving training efficiency. <!-- (Figure 6). --> ![](/images/DAPO-2.png)
1. Token-Level Policy Gradient Loss: Shifts from sample-level to token-level loss calculation, balancing the influence of long and short responses. This prevents low-quality, overly long generations and stabilizes training. <!-- (Figure 4). --> ![](/images/DAPO-3.png)
1. Overlong Reward Shaping: Introduces soft punishment for truncated responses to reduce reward noise. Instead of harsh penalties, it uses a length-aware function to guide models toward optimal response lengths. <!-- (Figure 5). --> ![](/images/DAPO-4.png)

#### Dataset and Implementation
The DAPO-Math-17K dataset transforms math problems into integer answers for reliable rule-based reward signals. The system is built on the verl framework, with training details including AdamW optimization, group reward normalization, and dynamic sampling hyperparameters.

#### Experiments and Results
- AIME 2024 Performance: DAPO achieves 50 points on AIME with Qwen2.5-32B, outperforming DeepSeek-R1-Zero-Qwen-32B (47 points) with half the training steps (Figure 1).
- Ablation Study: Each technique contributes significantly: Overlong Filtering (+6), Clip-Higher (+2), Soft Overlong Punishment (+3), Token-Level Loss (+1), and Dynamic Sampling (+8) (Table 1).
- Training Dynamics: Metrics like response length, reward, and entropy show stable improvement, with Clip-Higher specifically combating entropy collapse (Figure 7).

### Dr. GRPO
**Paper**: Liu, Zichen, et al. Understanding R1-Zero-Like Training: A Critical Perspective. arXiv:2503.20783, arXiv, 26 Mar. 2025. arXiv.org, https://doi.org/10.48550/arXiv.2503.20783.

This paper claims that:
- DeepSeek-V3-Base already exhibit “Aha moment”, while Qwen2.5  base models demonstrate strong reasoning capabilities even without  prompt templates, suggesting potential pretraining biases.
- There is an optimization bias in Group Relative Policy Optimization  (GRPO), which artificially increases response length (especially for incorrect outputs) during training.

And Dr. GRPO is an unbiased optimization method that improves token efficiency while maintaining reasoning performance.

![](/images/Dr.GRPO.png)

Key Insights on Base Models  
1. Template Impact on Question-Answering Ability
   - Base models like Llama and DeepSeek require prompt templates (e.g., R1 template) to elicit question-answering behavior, while Qwen2.5 models excel without templates. This suggests Qwen2.5 might be pretrained on question-answer pairs, acting like supervised fine-tuned (SFT) models even in their base form.  
   - For example, Qwen2.5-Math-7B achieves 69.0% accuracy on MATH500 without templates, outperforming traditional prompting methods.  

2. Preexisting "Aha Moment" in Base Models
   - The "Aha moment" (self-reflection behaviors) often attributed to RL emergence is already present in base models like DeepSeek-V3-Base. Experiments show these models generate self-reflection keywords (e.g., "Aha," "wait") in responses to math problems without RL tuning.  


Analysis of Reinforcement Learning  
1. Biases in GRPO  
   - GRPO introduces two key biases:  
     - Response-length bias: Dividing by response length ($|o_i|$) penalizes short correct responses and favors longer incorrect ones.  
     - Question-difficulty bias: Normalizing by reward standard deviation prioritizes easy/hard questions, skewing optimization.  
   - These biases lead to unnecessarily long incorrect responses, as observed in training dynamics.  

2. Dr. GRPO: Unbiased Optimization  
   - The authors propose Dr. GRPO, which removes $|o_i|$ and standard deviation normalization from GRPO. This fixes the biases, improving token efficiency while maintaining reasoning performance.  
   - Experiments show Dr. GRPO reduces the length of incorrect responses and matches the accuracy of GRPO with fewer tokens.  

### Group-in-Group Policy Optimization (GiGPO)
Nested group comparisons to handle hierarchical decision-making, suitable for complex tasks with multiple levels of abstraction.

**Paper**: Feng, Lang, et al. Group-in-Group Policy Optimization for LLM Agent Training. arXiv:2505.10978, arXiv, 16 May 2025. arXiv.org, https://doi.org/10.48550/arXiv.2505.10978.

![](/images/GiGPO.png)


## **III. Optimization without Reward Model**  
### **Direct Preference Optimization (DPO)**  
Direct Preference Optimization (DPO) is a novel training method for large language models that directly optimizes for human preferences without requiring a separate reward model. DPO eliminates the need for a reward model by directly optimizing a policy using pairwise preference data. It minimizes the KL divergence between the policy and a reference model while maximizing the Bradley-Terry loss.
![](/images/DPO.png)

**Paper**: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290). 

#### Core Innovation
DPO derives a closed-form solution that directly optimizes the policy using preference data. The key insight is that the optimal policy under the RLHF objective can be expressed analytically in terms of the reward function and reference policy.

The DPO objective is based on the Bradley-Terry preference model, the optimal RLHF policy $π^∗$ under the Bradley-Terry model satisfies the preference model::

$$P(y_w \succ y_l | x) = \frac{1}{1 + \exp(\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} - \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)})}$$

Where:
- $y_w$ is the preferred (winning) response
- $y_l$ is the less preferred (losing) response
- $x$ is the input prompt
- $\pi_\theta$ is the policy being optimized
- $\pi_{ref}$ is the reference policy
- $\beta$ is a temperature parameter

For a static dataset of comparisons $\mathcal{D} = \left\{ x^{(i)}, y_w^{(i)}, y_l^{(i)} \right\}_{i=1}^N$, the reward modeling approach works by defining:
$$
\mathcal{L}_R(r_\phi, \mathcal{D}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right]
$$

Analogous to reward modeling approach, DPO's policy objective becomes:
$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$
The gradient with respect to the parameters $\theta$ can be written as:
$$
\begin{split}
\nabla_\theta \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) &= \\
&- \beta \mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \bigg[ \underbrace{\sigma\bigl(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\bigr)}_{\text{higher weight when reward estimate is wrong}} \, \bigg[ \underbrace{\nabla_\theta \log \pi(y_w \mid x)}_{\text{increase likelihood of } y_w} - \underbrace{\nabla_\theta \log \pi(y_l \mid x)}_{\text{decrease likelihood of } y_l} \bigg] \bigg],
\end{split}
$$

where $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ is the reward implicitly defined by the language model $\pi_\theta$ and reference model $\pi_{\text{ref}}$. 

#### A mechanistic understanding of DPO
- The gradient of the loss function $\mathcal{L}_{\text{DPO}}$ increases the likelihood of the preferred completions $y_w$ and decreases the likelihood of dispreferred completions $y_l$. 
- The examples are weighed by how much higher the implicit reward model $\hat{r}_\theta$ rates the dispreferred completions, scaled by $\beta$, i.e, how incorrectly the implicit reward model orders the completions, accounting for the strength of the KL constraint. 
- The papers's experiments suggest the importance of this weighting, as a naïve version of this method without the weighting coefficient can cause the language model to degenerate.

<!-- $$L_{DPO} = -\mathbb{E}_{(x,y_w,y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$ -->

#### Training Process

1. **Data Collection**: Gather preference pairs $(x, y_w, y_l)$ where $y_w$ is preferred over $y_l$ for prompt $x$

2. **Direct Optimization**: Minimize the DPO loss $\mathcal{L}_{\text{DPO}}$
    

3. **Regularization**: The KL divergence constraint from RLHF is implicitly maintained through the reference policy terms

#### Advantages of DPO
- **Simplicity**
    - Eliminates the need for reward model training
    - Reduces the training pipeline from 3 stages to 2 stages
    - Avoids the complexities of reinforcement learning
- **Stability**
    - More stable training compared to PPO-based RLHF
    - No issues with reward model overoptimization
    - Direct gradient-based optimization
- **Efficiency**
    - Requires less computational resources
    - Faster convergence
    - Easier hyperparameter tuning

#### Practical Implementation

```python
# Simplified DPO loss computation
def dpo_loss(policy_chosen_logps, policy_rejected_logps, 
             reference_chosen_logps, reference_rejected_logps, beta=0.1):
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps
    
    logits = beta * (policy_logratios - reference_logratios)
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    
    return loss
```

#### Limitations and Considerations

1. **Data Quality**: Heavily dependent on high-quality preference data
2. **Distribution Shift**: May struggle with significant shifts from reference policy
3. **Preference Complexity**: Works best with clear preference distinctions
4. **Hyperparameter Sensitivity**: The β parameter requires careful tuning


Comparison with RLHF

| Aspect | RLHF | DPO |
|--------|------|-----|
| Complexity | High (3 stages) | Medium (2 stages) |
| Stability | Can be unstable | More stable |
| Computational Cost | High | Lower |
| Flexibility | High | Moderate |

### **Online DPO - Direct Language Model Alignment from Online AI Feedback**
**Paper**: Guo, Shangmin, et al. Direct Language Model Alignment from Online AI Feedback. arXiv:2402.04792, arXiv, 29 Feb. 2024. arXiv.org, https://doi.org/10.48550/arXiv.2402.04792.

Direct Alignment from Preferences (DAP) methods like DPO have emerged as efficient alternatives to RLHF, but they rely on pre-collected offline preference data. This leads to two key issues:
1. **Offline Feedback**: Preferences are static and not updated during training.
2. **Off-Policy Learning**: Responses in the dataset are generated by a different model, causing distribution shift as the target model evolves.

The proposed Online AI Feedback (OAIF) framework makes DAP methods online and on-policy by:
1. Sampling two responses from the current model for each prompt.
2. Using an LLM annotator (e.g., PaLM 2) to provide real-time preference feedback by choosing the better response.
3. Updating the model using standard DAP losses (DPO, IPO, SLiC) with this online feedback.

Key advantages:
- Avoids distribution shift by using on-policy generations.
- Eliminates the need for a separate Reward Model (RM), unlike RLHF.
- Enables controllable feedback via prompt engineering for the LLM annotator.

#### Experiments and Results
1. **Effectiveness vs. Offline DAP**:
   - Online DAP methods (DPO, IPO, SLiC) achieved an average 66% win rate over their offline counterparts in human evaluations.
   - Online DPO outperformed SFT baselines, RLHF, and RLAIF 58% of the time on the TL;DR task.

2. **Generalization to Other DAP Methods**:
   - OAIF improved all three DAP methods, with online SLiC showing a 71.43% win rate over offline SLiC in TL;DR.

3. **Comparison with RLHF/RLAIF**:
   - Online DPO was preferred 58% of the time in 4-way comparisons (vs. offline DPO, RLAIF, RLHF).
   - RLHF relies on static RMs, which struggle as the model evolves, while OAIF's LLM annotator adapts dynamically.

4. **Controllability via Prompts**:
   - Instructing the LLM annotator to prefer shorter responses reduced average token length from ~120 to ~40, while maintaining quality above SFT baselines.

5. **Impact of Annotator Size**:
   - Larger annotators (e.g., PaLM 2-L) improved performance, but even smaller annotators (PaLM 2-XS) outperformed RLHF in some cases.

#### Conclusion
OAIF addresses the offline and off-policy limitations of DAP methods, achieving better alignment with reduced human annotation. The approach paves the way for scalable LLM alignment using AI feedback, with potential for real-time user adaptation and qualitative objective control.

#### Table 1: Method Comparison
| Method         | No RM Needed | On-Policy | Online Feedback |
|----------------|--------------|-----------|-----------------|
| Offline DAP    | ✓            | ✗         | ✗               |
| RLHF/RLAIF     | ✗            | ✓         | ✓               |
| OAIF (Proposed)| ✓            | ✓         | ✓               |

### **TDPO: Token-level Direct Preference Optimization**
DPO optimize models at the sentence level, evaluating full responses. However, LLMs generate text token-by-token in an auto-regressive manner, which creates a mismatch between evaluation and generation processes. DPO uses KL divergence to constrain models to a reference LLM but struggles with divergence efficiency: the KL divergence of dispreferred responses grows too quickly, limiting diversity. This motivates the need for a more granular, token-level optimization approach.

TDPO optimizes token-level preferences to improve sequence generation quality, addressing limitations of instance-level DPO in long-chain reasoning.  

**Paper**: [Token-level Direct Preference Optimization](https://arxiv.org/abs/2404.11999).  

![](/images/TDPO.png)

TDPO models text generation as an MDP, where:
- State \( s_t = [x, y^{\lt t}] \) (prompt + partial response)
- Action \( a_t = y^t \) (next token)
- Reward \( R_t = R(s_t, a_t) \) (token-wise reward)

The objective function combines the advantage function of a reference model with forward KL divergence:
\[
\max_{\pi_\theta} \mathbb{E}\left[ A_{\pi_{ref}}(s_t, z) - \beta D_{KL}(\pi_\theta(\cdot|s_t) \| \pi_{ref}(\cdot|s_t)) \right]
\]
where \( A_{\pi_{ref}} \) is the advantage function, and \( \beta \) weights the KL penalty.

TDPO transforms the sentence-level Bradley-Terry model into a token-level preference model, relating it to the Regret Preference Model. The key is expressing human preference probability as:
\[
P_{BT}(y_1 \succ y_2|x) = \sigma(u(x, y_1, y_2) - \delta(x, y_1, y_2))
\]
where \( u \) is the reward difference from DPO, and \( \delta \) is the weighted SeqKL difference between responses.

The final loss function for TDPO (version TDPO2) is:
\[
\mathcal{L}_{TDPO2} = -\mathbb{E}\left[ \log\sigma\left(u(x, y_w, y_l) - \alpha\delta_2(x, y_w, y_l)\right) \right]
\]
Here, \( \alpha \) controls the balance of SeqKL divergence, and \( \delta_2 \) incorporates a stop-gradient operator to stabilize training.

#### Key Contributions
1. **Token-Level Optimization**: TDPO introduces sequential KL divergence (SeqKL) to monitor divergence at each token, addressing the inefficiency of sentence-level KL in DPO.
2. **Forward KL Constraints**: By incorporating forward KL divergence for each token, TDPO balances alignment with generation diversity, overcoming the mode-seeking limitation of reverse KL in DPO.
3. **Bradley-Terry Model Adaptation**: TDPO reformulates the Bradley-Terry model (for pairwise preference) at the token level, connecting it to the Regret Preference Model for better reward representation.
4. **Experimental Superiority**: TDPO outperforms DPO and PPO-based RLHF in balancing alignment and diversity across sentiment generation, dialogue, and multi-turn reasoning tasks.

### **Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning**  
Focuses on step-by-step reasoning by isolating errors in individual steps of long chains. It uses tree search to generate step-level preference data and enhances mathematical reasoning.  
**Paper**: Jia, J., et al. (2024). [Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs](https://arxiv.org/abs/2406.18629).  



### **Identity Preference Optimization (IPO)**
RLHF and DPO can overfit deterministic preferences because the logit transformation (Ψ) amplifies small preference differences near 1, weakening KL regularization. This leads to policies deviating from the reference policy, even with large regularization parameters.

ΨPO is defined as maximizing a non-linear function of preference probabilities (Ψ) balanced by KL regularization to a reference policy. RLHF and DPO are shown to be special cases when Ψ is the logit function, relying on the Bradley-Terry model. 

Identity-PO (IPO) is an approach setting Ψ to the identity function, bypassing the Bradley-Terry model. IPO optimizes total preferences directly, maintaining effective regularization even with deterministic preferences. An empirical sampled loss is derived for practical implementation, avoiding reward modeling.

**Paper**: [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036).  

IPO Loss function:

$$
\mathbb{E}_{(y_w, y_l) \sim D} \left[ \left( h_\pi(y_w, y_l) - \frac{\tau^{-1}}{2} \right)^2 \right]
$$

IPO learns from preferences dataset simply by regressing the gap between log-likelihood ratios $\log(\pi(y_w)/\pi(y_l))$ and $\log(\pi_{\text{ref}}(y_w)/\pi_{\text{ref}}(y_l))$ to $\frac{\tau^{-1}}{2}$. 

So the weaker the regularisation becomes, the higher would be the log-likelihood ratio of $y_w$ to $y_l$. 

In other words IPO, unlike DPO, always regularizes its solution towards $\pi_{\text{ref}}$ by controlling the gap between the log-likelihood ratios $\log(\pi(y_w)/\pi(y_l))$ and $\log(\pi_{\text{ref}}(y_w)/\pi_{\text{ref}}(y_l))$, thus avoiding the over-fitting to the preference dataset.


```python
# ... calculate logits the same as DPO
# https://github.com/huggingface/trl/blob/2c49300910e55fd7482ad80019feee4cdaaf272c/trl/trainer/dpo_trainer.py#L974
# eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
losses = (logits - 1 / (2 * self.beta)) ** 2
```

### SPPO - Self-Play Preference Optimization for Language Model Alignment

**Paper**: Wu, Yue, et al. Self-Play Preference Optimization for Language Model Alignment. arXiv:2405.00675, arXiv, 4 Oct. 2024. arXiv.org, https://doi.org/10.48550/arXiv.2405.00675.

### **DMPO: Direct Multi-Turn Preference Optimization**  
Extends DPO to multi-turn dialogue by optimizing preferences across conversation history, improving coherence and relevance in multi-step interactions.  

**Paper**: Shi, Wentao, et al. Direct Multi-Turn Preference Optimization for Language Agents. arXiv:2406.14868, arXiv, 23 Feb. 2025. arXiv.org, https://doi.org/10.48550/arXiv.2406.14868.

### WPO - Enhancing RLHF with Weighted Preference Optimization

## **IV. Reference-Free Optimization** 
### **SimPO: Simple Preference Optimization with a Reference-Free Reward**
SimPO aligns the reward function with the generation metric, eliminates the need for a reference model, and introduces a target reward margin to enhance performance.

![](/images/SimPO.png)

In DPO, for any triple $(x, y_w, y_l)$, satisfying the reward ranking $r(x, y_w) > r(x, y_l)$ does not necessarily gaurantee the likelihood ranking $p_\theta(y_w \mid x) > p_\theta(y_l \mid x)$. The paper found that only roughly 50\% of the triples from a held-out set satisfy this condition when trained with DPO.
 <!-- (see Figure 4b)   -->

**Paper**: [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734).  


**Length-Normalized Reward formulation**:
Replacing the reward formulation in DPO with $p_\theta$ in the average log likelihood
$$
p_\theta(y \mid x) = \frac{1}{|y|} \log \pi_\theta(y \mid x) = \frac{1}{|y|} \sum_{i=1}^{|y|} \log \pi_\theta(y_i \mid x, y_{\lt i}).
$$
so that it aligns with the likelihood metric that guides generation, results in a length-normalized reward:
$$
r_{\text{SimPO}}(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y \mid x) = \frac{\beta}{|y|} \sum_{i=1}^{|y|} \log \pi_\theta(y_i \mid x, y_{\lt i})
$$

where $\beta$ is a constant that controls the scaling of the reward difference.

This reward eliminates the need for a reference model, enhancing memory and computational efficiency.

Removing the length normalization term from the reward formulation results in a bias toward generating longer but lower-quality sequences.

**Target Reward Margin**:
SimPO incorporates a target margin $\gamma$ into the Bradley-Terry objective to ensure the reward difference between winning ($y_w$) and losing ($y_l$) responses exceeds $\gamma$:  
$$
  \mathcal{L}_{\text{SimPO}} = -\mathbb{E}\left[\log \sigma\left(\frac{\beta}{|y_w|}\log\pi_{\theta}(y_w|x) - \frac{\beta}{|y_l|}\log\pi_{\theta}(y_l|x) - \gamma\right)\right]
$$  

This margin enhances the model’s ability to distinguish between high-quality and low-quality responses, improving generalization.  


**Implementation**:
https://docs.pytorch.org/torchtune/0.3/_modules/torchtune/rlhf/loss/dpo.html#SimPOLoss.forward
```python
def simpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    simpo_gamma,
    beta
):
    logits = (policy_chosen_logps - policy_rejected_logps)
    gamma_logratios = simpo_gamma / beta
    logits = logits - gamma_logratios
    losses = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(beta * logits) * label_smoothing
    )
    return losses
```

### **Odds Ratio Preference Optimization (ORPO)**
ORPO (Odds Ratio Preference Optimization) combines SFT and preference optimization in a single monolithic training stage, eliminating reference model requirements. This approach streamlines training while maintaining competitive performance across multiple benchmarks.

**Paper**: Hong, S., et al. (2024). [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691).

Existing methods like RLHF and DPO often require:
- A supervised fine-tuning (SFT) warm-up stage.
- A reference model for comparison.
- Complex multi-stage processes, which are resource-intensive.

The core insight: SFT can be enhanced to directly incorporate preference alignment by penalizing undesired generation styles, eliminating the need for extra stages or reference models.

#### **ORPO Algorithm**
![](/images/ORPO.png)

ORPO integrates preference alignment into SFT using an **odds ratio** to contrast favored (\(y_w\)) and disfavored (\(y_l\)) responses. The key components are:
- **Objective Function**: \(\mathcal{L}_{ORPO} = \mathcal{L}_{SFT} + \lambda \cdot \mathcal{L}_{OR}\), where:
  - \(\mathcal{L}_{SFT}\) is the standard negative log-likelihood (NLL) loss for SFT.
  - \(\mathcal{L}_{OR}\) is a new loss term that maximizes the odds ratio between \(y_w\) and \(y_l\), defined as:
    \[
    \mathcal{L}_{OR} = -\log \sigma\left(\log \frac{odds_\theta(y_w|x)}{odds_\theta(y_l|x)}\right)
    \]
    where \(odds_\theta(y|x) = \frac{P_\theta(y|x)}{1-P_\theta(y|x)}\) and \(\sigma\) is the sigmoid function.
- **Gradient Analysis**: The gradient of \(\mathcal{L}_{OR}\) dynamically penalizes disfavored responses, accelerating adaptation to desired styles while preserving domain knowledge from SFT.

Implementation from https://github.com/huggingface/trl/blob/2c49300910e55fd7482ad80019feee4cdaaf272c/trl/trainer/orpo_trainer.py#L623
```python
def odds_ratio_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute ORPO's odds ratio (OR) loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the ORPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
            The log odds ratio of the chosen responses over the rejected responses ratio for logging purposes.
            The `log(sigmoid(log_odds_chosen))` for logging purposes.
        """

        # Derived from Eqs. (4) and (7) from https://huggingface.co/papers/2403.07691 by using log identities and exp(log(P(y|x)) = P(y|x)
        log_odds = (policy_chosen_logps - policy_rejected_logps) - (
            torch.log1p(-torch.exp(policy_chosen_logps)) - torch.log1p(-torch.exp(policy_rejected_logps))
        )
        ratio = F.logsigmoid(log_odds)
        losses = self.beta * ratio

        chosen_rewards = self.beta * (policy_chosen_logps.to(self.accelerator.device)).detach()
        rejected_rewards = self.beta * (policy_rejected_logps.to(self.accelerator.device)).detach()

        return losses, chosen_rewards, rejected_rewards, torch.mean(ratio), torch.mean(log_odds)

```

Advantages over Existing Methods**
- Monolithic Design: ORPO performs preference alignment in a single stage during SFT, unlike RLHF/DPO’s multi-stage workflows.
- No Reference Model: Avoids the need for a frozen SFT model, reducing memory usage and computational cost (half the forward passes per batch).
- Stability with Odds Ratio: Compared to probability ratios, the odds ratio provides a milder penalty, preventing over-suppression of disfavored responses (Figure 6).


## **V. Non-preference-based methods**
### **KTO - Model Alignment as Prospect Theoretic Optimization**
The paper introduces **Kahneman-Tversky Optimization (KTO)**, a approach maximizes the utility of generations instead of maximizing the log-likelihood of preferences(as DPO does). 

Theoretically, KTO align LLM with human preferences by framing alignment as a problem of prospect theoretic optimization. Drawing from Kahneman and Tversky’s prospect theory, which describes how humans make decisions under uncertainty with biases like loss aversion, the authors argue that existing alignment methods (e.g., DPO, PPO) implicitly incorporate these biases. They formalize these as **Human-Aware Losses (HALOs)** and propose KTO as a principled HALO that directly maximizes human utility rather than the log-likelihood of preferences.


**Paper**: [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306).  

**KTO Method**:
- Derives a HALO using Kahneman-Tversky’s value function, replacing exponents with logistic functions for stability.
- Requires only binary feedback (desirable/undesirable) instead of preference pairs, making data collection cheaper and more scalable.
- Introduces hyperparameters β (risk aversion) and λ_D/λ_U (loss aversion for desirable/undesirable outputs) to model human decision biases.

**Prospect Theory in Alignment**:
- **Value Function**: Models human utility relative to a reference point, with concavity in gains and steeper slopes in losses (loss aversion).
- **Reference Point**: In KTO, the reference is the KL divergence between the current policy and the reference model, estimated via microbatch mismatches for stability.

**Loss Function**:

$\lambda_y$ denotes $\lambda_D$ ($\lambda_U$) when $y$ is desirable(undesirable) respectively, the default KTO loss is:
$$
\mathcal{L}_{\text{KTO}}(\pi_\theta, \pi_{\text{ref}}) \triangleq \mathbb{E}_{x,y \sim D} \left[ \lambda_y - v(x,y) \right] \tag{8}
$$
where
$$
r_\theta(x,y) = \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)},
$$
$$
z_0 = \text{KL}\bigl(\pi_\theta(y'|x) \big\| \pi_{\text{ref}}(y'|x)\bigr),
$$
$$
v(x,y) = 
\begin{cases} 
\lambda_D \sigma\bigl(\beta(r_\theta(x,y) - z_0)\bigr) & \text{if } y \sim y_{\text{desirable}}|x, \\
\lambda_U \sigma\bigl(\beta(z_0 - r_\theta(x,y))\bigr) & \text{if } y \sim y_{\text{undesirable}}|x.
\end{cases}
$$

Implementation from https://github.com/huggingface/trl/blob/2c49300910e55fd7482ad80019feee4cdaaf272c/trl/trainer/kto_trainer.py#L1090:
```python
def kto_loss(
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ):
    kl = (policy_KL_logps - reference_KL_logps).mean().detach()
    # Chosen losses
    if policy_chosen_logps.shape[0] != 0 or reference_chosen_logps.shape[0] != 0:
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        chosen_losses = 1 - F.sigmoid(beta * (chosen_logratios - kl))
        chosen_rewards = beta * chosen_logratios.detach()
    else:
        chosen_losses = torch.Tensor([])
        chosen_rewards = torch.Tensor([])
        
    # Rejected losses
    if policy_rejected_logps.shape[0] != 0 or reference_rejected_logps.shape[0] != 0:
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        rejected_losses = 1 - F.sigmoid(beta * (kl - rejected_logratios))
        rejected_rewards = beta * rejected_logratios.detach()
    else:
        rejected_losses = torch.Tensor([])
        rejected_rewards = torch.Tensor([])
        
    losses = torch.cat(
        (chosen_losses, rejected_losses),
        0,
    )
    return losses

```



## **VI. Others**


### **SPO - A Minimaximalist Approach to Reinforcement Learning from Human Feedback**  
Uses self-play to generate preference data, where models compete to maximize rewards. It employs a minimax approach to balance exploration and exploitation.  
**Paper**: OpenAI. (2024). [A Minimaximalist Approach to Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2401.04056).  

## **VII. Key Trends and Challenges**  
1. **Scalability**: Methods like RLHF and DPO face challenges with large models (e.g., 405B parameters) due to memory and computational costs. Frameworks like LlamaRL address this via asynchronous distributed training .  
2. **Data Efficiency**: SimPO and ORPO reduce reliance on reference models and pairwise data, while Step-DPO and KTO focus on fine-grained optimization .  
3. **Generalization**: SPO and DMPO aim to improve generalization across tasks (e.g., dialogue, math) by leveraging self-play and multi-turn optimization .  
4. **Theoretical Foundations**: KTO and SimPO provide theoretical insights into preference learning, linking RL to prospect theory and information theory .  

## **VIII. Future Directions**  
- **Multi-Modality**: IPO and DMPO highlight the need for alignment across text, video, and dialogue.  
- **Self-Supervised Learning**: RLAIF and SPO explore AI-generated feedback to reduce human labeling.  
- **Efficiency**: LlamaRL and SimPO demonstrate advancements in distributed training and reward design.  

For the latest implementations, refer to repositories like [LlamaRL](https://arxiv.org/abs/2505.24034) and [SimPO](https://paperswithcode.com/paper/simpo-simple-preference-optimization-with-a).