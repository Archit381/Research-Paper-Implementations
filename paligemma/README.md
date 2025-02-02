# PaliGemma: A versatile 3B VLM for transfer

![images](https://raw.githubusercontent.com/Archit381/Research-Paper-Implementations/refs/heads/main/paligemma/assets/paligemma_architecture.png)

Paligemma is a Vision Language Model (VLM) that uses SigLIP (improved form of CLIP) and Gemma Decoder Block which can be used in following tasks:

 - Visual Question and Answering
 - Image Captioning
 - Object Detection ( Not implemented in this repo )


### Some insights into CLIP (Contrastive Language-Image Pretraining )

A contrastive vision encoder is something that takes an image as input and converts it into and embedding of each pixel. CLIP uses a VIT like transformer to extact the image features and a causal language model to get the textual features. Both the image and textual embeddings are then projected to a latent space with identical dimension and concatenated.

 - The main objective of contrastive learning is to ensure that the dot product between the embedding of an image and its corresponding text is high, while the dot product between unrelated text and image embeddings is low.
 
 - The loss function will be designed to encourage the model to minimize the dot products between non-related pairs and maximize the dot produce between related pairs. CLIP uses Cross-Entropy loss

    ### Problem with CLIP: Introducing SigLIP

    In CLIP, cross entropy and softmax require calculating max value for each row and column making it difficult to parallelize the computation efficiently.

    To help with this a sigmoid based binary classification approach is introduced where dot-product between and image and text embedding is treated as an independent binary classification task (related pairs are 1 and others are 0). This eliminates the need for softmax normalization step.


Once the image embeddings are generated from the SigLIP the image embeddings go through linear projection to be compatible with the embeddings shape of textual embeddings. These 2 embeddings are then concatenated and fed into the decoder block to generate the final textual output.

## References: 

 - https://www.youtube.com/watch?v=vAmKB7iPkWw
 - https://github.com/hkproj/pytorch-paligemma
 - https://arxiv.org/abs/2407.07726
