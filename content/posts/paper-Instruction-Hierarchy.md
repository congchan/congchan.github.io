---
title: Paper Reading - The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions
date: 2024-04-20
mathjax: true
tags: ['Readings', '2024', 'LLM', Alignment']
---

### Summary of "The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions"  
Wallace, Eric, et al. The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions. arXiv:2404.13208, arXiv, 19 Apr. 2024. arXiv.org, http://arxiv.org/abs/2404.13208.


#### 1. **Problem Statement**  
Modern large language models (LLMs) are vulnerable to attacks like prompt injections and jailbreaks because they treat system prompts, user messages, and third-party inputs (e.g., tool outputs) as equal in priority. This allows adversaries to override intended instructions, leading to risks such as data exfiltration or unauthorized actions .  

#### 2. **The Instruction Hierarchy Framework**  
The authors propose an instruction hierarchy to define priority levels for different input types:  
- **Highest priority**: System messages (developer-provided instructions)  
- **Medium priority**: User messages  
- **Lowest priority**: Third-party content (e.g., tool outputs, web results) .  

When instructions conflict, LLMs should prioritize higher-level instructions and ignore or refuse lower-privileged, misaligned commands. For example, a system message instructing an LLM to act as an email assistant should override a user’s attempt to inject a command to forward all emails .  

#### 3. **Training Approach**  
- **Aligned Instructions**: Use synthetic data generation to decompose complex requests (e.g., "write a 20-line poem in Spanish") into sub-instructions at different hierarchy levels, training models to compose responses.  
- **Misaligned Instructions**: Employ "context ignorance" to train models to ignore lower-privileged instructions that conflict with higher ones. For example, if a tool output contains a prompt injection, the model should respond as if the injection did not exist .  

#### 4. **Experimental Results**  
- **Robustness Improvements**: The method significantly enhances defense against various attacks:  
  - System prompt extraction: +63% improvement .  
  - Jailbreak resistance: +30% improvement .  
  - Indirect prompt injections via browsing: from 32.8% to 89.6% robustness .  
- **Generalization**: The model shows strong performance against unseen attacks, such as password extraction and tool-based injections, without compromising core capabilities (e.g., TriviaQA accuracy remains comparable) .  
- **Trade-offs**: Some "over-refusal" of benign queries occurs but is manageable with further data collection .  

#### 5. **Future Directions**  
- Refine conflict handling for tool outputs and multi-modal inputs (e.g., images, audio) .  
- Explore architectural changes (e.g., specialized embeddings for different priority levels) .  
- Enhance adversarial training to address remaining vulnerabilities .  

#### 6. **Key Contributions**  
- A formalized instruction hierarchy to prioritize security-critical instructions.  
- Automated data generation methods to teach hierarchical behavior.  
- Empirical evidence that the approach boosts LLM robustness without sacrificing general capabilities .
