---
title: Scaling Laws for Neural Language Models
date: 2021-12-19
mathjax: true
author: "Cong Chan"
tags: ['2020', 'Pre-Trained Models']
---

Kaplan, Jared, et al. ‘Scaling Laws for Neural Language Models’. *arXiv:2001.08361 [Cs, Stat]*, Jan. 2020. *arXiv.org*, [http://arxiv.org/abs/2001.08361](http://arxiv.org/abs/2001.08361).

# TL:DR

key findings for Transformer language models are are as follows:

- **Performance depends strongly on scale, weakly on model shape**: Model performance depends most strongly on scale, which consists of three factors: the number of model parameters N (excluding embeddings), the size of the dataset D, and the amount of compute C used for training. Within reasonable limits, performance depends very weakly on other architectural hyperparameters such as depth vs. width. (Section 3)
- **Smooth power laws**: Performance has a power-law relationship with each of the three scale factors N, D, C when not bottlenecked by the other two, with trends spanning more than six orders of magnitude (see Figure 1). We observe no signs of deviation from these trends on the upper end, though performance must flatten out eventually before reaching zero loss. (Section 3)

![](/images/papers/paper14.png)

- **Universality of overfitting**: Performance improves predictably as long as we scale up N and D in tandem, but enters a regime of diminishing returns if either N or D is held fixed while the other increases. The performance penalty depends predictably on the ratio N 0.74/D, meaning that every time we increase the model size 8x, we only need to increase the data by roughly 5x to avoid a penalty. (Section 4)
- **Universality of training**: Training curves follow predictable power-laws whose parameters are roughly independent of the model size. By extrapolating the early part of a training curve, we can roughly predict the loss that would be achieved if we trained for much longer. (Section 5)
- **Transfer improves with test performance**: When we evaluate models on text with a different distribution than they were trained on, the results are strongly correlated to those on the training validation set with a roughly constant offset in the loss – in other words, transfer to a different distribution incurs a constant penalty but otherwise improves roughly in line with performance on the training set. (Section 3.2.2)
- **Sample efficiency**: Large models are more sample-efficient than small models, reaching the same level of performance with fewer optimization steps (Figure 2) and using fewer data points (Figure 4).

![](/images/papers/paper14-1.png)

![](/images/papers/paper14-2.png)


- **Convergence is inefficient**: When working within a fixed compute budget C but without any other restrictions on the model size N or available data D, we attain optimal performance by training very large models and stopping significantly short of convergence (see Figure 3). Maximally compute-efficient training would therefore be far more sample efficient than one might expect based on training small models to convergence, with data requirements growing very slowly as D ∼ C0.27 with training compute. (Section 6)
![](/images/papers/paper14-3.png)

- **Optimal batch size**: The ideal batch size for training these models is roughly a power of the loss only, and continues to be determinable by measuring the gradient noise scale [MKAT18]; it is roughly 1-2 million tokens at convergence for the largest models we can train. (Section 5.1)