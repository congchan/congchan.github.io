---
title: The Curious Case of Neural Text Degeneration
date: 2021-12-23
mathjax: true
author: "Cong Chan"
tags: ['2020', 'Language Models', 'Text Generation']
---

Holtzman, Ari, et al. The Curious Case of Neural Text Degeneration. arXiv:1904.09751, arXiv, 14 Feb. 2020. arXiv.org, http://arxiv.org/abs/1904.09751.

# Introduction

从语言模型生成文本（例如生成故事）的最佳解码策略是什么仍然是一个悬而未决的问题。违反直觉的经验观察是，即使使用似然作为训练目标可以为广泛的语言理解任务生成高质量的模型，但基于maximization-based decoding的解码方法（例如beam search）会导致退化（degeneration）——输出文本平淡无奇，不连贯，或陷入重复循环。

文本生成中的decoding strategy主要可以分为两大类：

- Argmax Decoding: 主要包括beam search, class-factored softmax等
- Stochastic Decoding: 主要包括temperature sampling, top-k sampling等

为了解决这个问题，提出了 **Nucleus Sampling（Top-p Sampling）**，这是一种简单但有效的方法，可以从神经语言模型中提取比以前的解码策略质量更高的文本。The key idea is to use the shape of the probability distribution to determine the set of tokens to be sampled from.

# Method

通过截断概率分布的不可靠尾部分布、从包含绝大多数概率质量的标记的dynamic nucleus中采样来避免文本退化。

![](/images/papers/paper19.png)

![](/images/papers/paper19-1.png)

![](/images/papers/paper19-2.png)

![](/images/papers/paper19-3.png)

![](/images/papers/paper19-4.png)

# 效果/Analysis/Findings

为了正确检查当前基于最大化和随机的解码方法，我们将这些方法中的每一种的生成与人类文本从几个方向（如可能性、多样性和重复）的分布进行了比较。

结果表明（1）对于开放式文本生成，maximization不是合适的解码目标，（2）当前最好的语言模型的概率分布有一个不可靠的长尾，需要在生成过程中截断，以及（3）Nucleus Sampling是目前最佳的解码策略，用于生成高质量的长文本——根据人类评估衡量——并且与人类编写的文本一样多样化。

![](/images/papers/paper19-5.png)

# 延伸阅读

- [https://zhuanlan.zhihu.com/p/68383015](https://zhuanlan.zhihu.com/p/68383015)