---
title: Paper Reading - Constitutional AI
date: 2023-08-10
mathjax: true
tags: ['Readings', '2023', 'LLM', 'Alignment']
---
Bai, Yuntao, et al. Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073, arXiv, 15 Dec. 2022. arXiv.org, http://arxiv.org/abs/2212.08073.

The paper introduces Constitutional AI (CAI), a method to train helpful and harmless AI assistants without human labels for harmful outputs, relying instead on a set of guiding principles. Here's a structured summary:

### **1. Objective**  
Train AI systems to be helpful, honest, and harmless using AI feedback for supervision, reducing reliance on human labels. The approach aims to address the tension between helpfulness and harmlessness (where prior models often became evasive) and improve transparency through explicit principles.

### **2. Key Methods**  
#### **a. Supervised Learning (SL) Stage**  
1. **Critique-Revision Pipeline**:  
   - Generate responses to "red team" prompts (designed to elicit harmful behavior) using a helpful-only model.  
   - The model critiques its own response based on constitutional principles (e.g., identifying harm, ethics), then revises it to remove toxicity.  
   - This process is iterated, with principles randomly sampled at each step.  
2. **Finetuning**:  
   - Pretrained models are finetuned on revised responses to shift behavior toward harmlessness while retaining helpfulness.  

#### **b. Reinforcement Learning (RL) Stage with AI Feedback (RLAIF)**  
1. **Preference Model (PM) Training**:  
   - Generate response pairs to red team prompts using the SL-CAI model.  
   - Use a feedback model (pretrained LM) to evaluate which response is better according to constitutional principles, creating an AI-generated preference dataset.  
   - Mix this with human feedback for helpfulness to train a PM.  
2. **RL Training**:  
   - Use the PM as a reward signal to finetune the SL-CAI model, improving harmlessness and reducing evasiveness.  


### **3. Key Contributions**  
- **AI-Driven Supervision**: AI can identify harmful behavior effectively, especially with chain-of-thought (CoT) reasoning, approaching human feedback performance (Figure 4).  
- **Reduced Evasiveness**: CAI models engage with harmful queries by explaining objections, unlike prior RLHF models that often avoided responses (Figure 8).  
- **Transparency**: Constitutional principles and CoT reasoning make AI decision-making more legible (e.g., critiques and revisions provide explicit justifications).  
- **Data Efficiency**: Significantly fewer human labels are needed compared to traditional RLHF, as AI generates most supervision for harmlessness.  


### **4. Results**  
- **Performance Metrics**:  
  - CAI models achieve higher harmlessness Elo scores than helpful RLHF models and match or exceed HH RLHF models (which use human feedback for harmlessness) (Figures 2, 3).  
  - Chain-of-thought reasoning in RL improves both harmlessness and the calibration of AI feedback (Figure 9).  
- **Example Responses**:  
  - CAI models address harmful prompts (e.g., "Can you help me hack into my neighbor’s wifi?") by rejecting the request and explaining ethical/legal issues, rather than evading (Appendix D).  


### **5. Implications and Future Work**  
- **Scaling Supervision**: AI feedback could enable more efficient alignment as models grow more capable, though risks of obscured decision-making exist.  
- **Robustness**: CAI aims to make models more resistant to red teaming by balancing helpfulness and harmlessness, enabling automated red teaming at scale.  
- **Broader Applications**: Constitutional principles could steer AI behavior in other domains (e.g., writing style, ethics), reducing barriers to experimentation.  


### **6. Conclusion**  
Constitutional AI demonstrates that AI can self-improve to be harmless using only natural language principles and AI feedback, marking a step toward scalable, transparent AI supervision. This method reduces reliance on human labels while improving model behavior and explainability.