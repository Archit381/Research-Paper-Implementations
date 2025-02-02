# Efficiently Modeling Long Sequences with Structured State Spaces

Since its introduction, Transformers have changed how feature representations are learned in deep learning models. The progress since has been immense where now transformers are the most reliable architecture to implement for a wide range of tasks involving image, text etc. The attention mechanism addressed the problem of traditional sequence models like simple RNNs, LSTMs not being able to handle long-range dependencies while also being slow in both training and inference due processing inputs in a linear manner.

## How did Transformers fix problems of Sequence Models ?

1. Tranformers introduced Self-Attention mechanism which basically computes attention   weights for each token by considering its dependence with all the other tokens regardless of their position. In simple words, instead of the prediction only being influenced by only a few tokens, self-attention allows all the tokens to contribute in the prediction.   

    <br>

    This helps in addressing the limitation of simple RNNs and LSTMs not being able to capture long-range dependencies between tokens. 

![images](https://raw.githubusercontent.com/Archit381/Research-Paper-Implementations/refs/heads/main/s4/assets/self_attention_diagram.png)

2. Unlike traditional sequence models that process tokens in a linear manner, in transformers all the tokens are processed in parallel which improve upon the training and inference time.

However, transformer pose a big problem of quadratic time complexity due to the self-attention mechanism. This makes it computationally expensive to train and becomes a bottle-neck since for each new token the attention for the entire sequence is calculated each time which leads to a high time complexity.

## Introducing State-Space Model (SSM)

SSM are designed to efficiently model long-range dependencies while also dealing with the quadratic time complexity of transformers.

SSMs are inspired from control theory where a system's state evolves over time based on a set of equations. The core idea is to maintain a hidden state that evolves dynamically across the sequence rather than computing direct pairwise interactions between all tokens like Transformers.

![images](https://raw.githubusercontent.com/Archit381/Research-Paper-Implementations/refs/heads/main/s4/assets/ssm_equations.png)

SSMs can be represented as both CNN and RNN, allowing for parallel training and fast inference

#### 1. Convolutional SSM

We observe that during the matrix multiplications in state equations, we can pre-compute some matrices (or parameters) that control how information flows over time. This pre-computed matrice is also called SSM convolution kernel or filter. 

![images](https://raw.githubusercontent.com/Archit381/Research-Paper-Implementations/refs/heads/main/s4/assets/cnn_eqn.png)

During training, these convolutions can be computed in parallel using FFT (fast fourier transform) and thus reduces the quadratic time complexity. These convolutions are applied in the Frequence Domain instead of step-by-step making them faster in training.

#### 2. Recurrence SSM

Training benefits from parallel approach, but when it comes to inference applying convolutions at each inference step is slow and inefficient. Plus, we can't really parallelize predictions since each step depends on the previous one but we can use the pre-computed matrix during inference to update the hidden state step by step. 

 - During the training, we process the entire sequence in parallel and treat it as a signal rather than a step-by-step process. But, since we need to compute token by token we need to discretize the continous signals for each time step.

 ![images](https://raw.githubusercontent.com/Archit381/Research-Paper-Implementations/refs/heads/main/s4/assets/discretization_eqn.png)

 Here, we introduce a new learnable parameter called step size (△).

  - We can then use the pre-computed matrix A and B and get much faster and efficient at inference.


### Problems with Regular SSMs

 1. Implementing just regular SSM will perform very poorly in practise due to the gradients getting scaled exponentially leading to a vanishing/exploding gradient problem. This means that SSMs forget long term-term dependencies.

 2. When we discretize the equations, a matrix exponential appears computing which is expensive and leads to high time complexity. If the step size changes, then we would need to recompute this matrix exponential making it inefficient for training.

### Introducing Structured State Space Models (S4)

Structured State Space for Sequences (S4), is a class of SSMs that can efficiently handle long sequences. It addresses both the issues of forgetting long-range dependencies and being computationally inefficient for long sequences.

#### 1. HiPPO Matrix

We introduce HIPPO matrix that provides a mathematically optimal way to compress history into a small state scale vector while retaining usefull information.

Lets consider a function f(t). We want to store all the past values of f(t) in a compact way rather than keeping all the values raw. Instead, we approximate it as a weighted
   sum of polynomials.

**`f(t)  = w1 * P1(t) + w2 * P2(t) + ...`**
    
here, P(t) are polynomials and w are weights that represent how much history alligns with each polynomial. This way f(t) is now summarized rather than storing all the past values.

**Note**: We only initialize the hippo matrix for A once.

#### 2. Diagonal Plus Low Rank

We assume a special structure for matrix A to be diagonal plus low rank. we assume A=Λ−PQ∗ where P,Q are small-rank vectors. A purely diagonal matrix is too restrictive and does'nt capture certain interactions. By adding a low-rank perturbation PQ*, we get a better approximation while still avoiding matrix inversion.

We also use the Woodbury matrix identity that simplifies the inversion of a matrix that has a special low-rank structure

The matrix exponentiation now reduces to a simple exponentiation of diagonal terms making the computations much cheaper. It also allows for efficient pre-computation of state transitions (parameters).


## References:

 - https://srush.github.io/annotated-s4/
 - https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state
 - https://arxiv.org/abs/2111.00396










