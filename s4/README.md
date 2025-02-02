# Efficiently Modeling Long Sequences with Structured State Spaces

Since its introduction, Transformers have changed how feature representations are learned in deep learning models. The progress since has been immense where now transformers are the most reliable architecture to implement for a wide range of tasks involving image, text etc. The attention mechanism addressed the problem of traditional sequence models like simple RNNs, LSTMs not being able to handle long-range dependencies while also being slow in both training and inference due processing inputs in a linear manner.

## How did Transformers fix problems of Sequence Models ?

1. Tranformers introduced Self-Attention mechanism which basically computes attention weights for each token by considering its dependence with all the other tokens regardless of their position. This helps in addressing the limitation of simple RNNs and LSTMs not being able to capture long-range dependencies between tokens. In simple words, instead of the prediction only being influenced by only a few tokens, self-attention allows all the tokens to contribute in the prediction.

![images](https://github.com/Archit381/Research-Paper-Implementations/tree/main/s4/assets/self_attention_diagram.png)

2. 








