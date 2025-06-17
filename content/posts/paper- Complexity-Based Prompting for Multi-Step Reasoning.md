---
title: Paper Reading -  Complexity-Based Prompting for Multi-Step Reasoning
date: 2023-04-19
mathjax: true
tags: ['ICLR', '2023', 'Large Language Model']
---

Tags: 2023, ICLR
Links: https://github.com/FranxYao/chain-of-thought-hub
Paper: Fu, Yao, et al. Complexity-Based Prompting for Multi-Step Reasoning. arXiv:2210.00720, arXiv, 30 Jan. 2023. arXiv.org, http://arxiv.org/abs/2210.00720.

# Motivation

Example selection is a central problem in the prompting literature.

For CoT prompting, example selection is further related to annotation efficiency, as CoT requires manually-annotated reasoning chains. Which reasoning examples make the most effective prompts. 

Propose complexity-based prompting, a simple and effective example selection scheme for multi-step reasoning. We show that prompts with higher reasoning complexity, i.e., chains with more reasoning steps, achieve substantially better performance on multistep reasoning tasks over strong baselines.

Further extend the complexity-based criteria from prompting (selecting inputs) to decoding (selecting outputs), where we sample multiple reasoning chains from the model, then choose the majority of generated answers from complex reasoning chains (over simple chains).

# Complexity-based prompting and consistency

vote over the top K complex chains

![](/images/papers/paper18.png)

Tasks: multi-step reasoning tasks, measured by solve rate (accuracy), is to predict the answer (typically a number) of a given math word problem via intermediate steps. Use math word problems, mathematical problems expressed in natural language, as our testbed.

The input is a stack of a few (often 8) CoT cases followed by a test question, then the language model continues generating an output CoT for the test question.

# Ref Reading

- Wang, Yizhong, et al. *Self-Instruct: Aligning Language Model with Self Generated Instructions*. arXiv:2212.10560, arXiv, 20 Dec. 2022. *arXiv.org*, [http://arxiv.org/abs/2212.10560](http://arxiv.org/abs/2212.10560).
- Lewkowycz, Aitor, et al. *Solving Quantitative Reasoning Problems with Language Models*. arXiv:2206.14858, arXiv, 30 June 2022. *arXiv.org*, [http://arxiv.org/abs/2206.14858](http://arxiv.org/abs/2206.14858).