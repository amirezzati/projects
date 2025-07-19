# Projects


## Deep Learning Projects

### [Retrieval-Augmented Generation for Psychological Applications](https://github.com/amirezzati/rag-psychology)

This project leverages Retrieval-Augmented Generation (RAG) to answer psychological questions and classify mental disorders based on renowned DSM-5 book. By combining the power of large language models with domain-specific datasets, the model provides accurate and relevant responses to specialized queries in the field of psychology.


### [Fine-Tuning of Stable Diffusion for Custom Concept Generation](https://github.com/amirezzati/deep-learning/blob/main/homeworks/HW5/pract/StableDiffusion.ipynb)

This project demonstrates how to adapt a large-scale diffusion model to capture and reproduce a novel visual concept from just a handful of examples. By fine-tuning only the core generative backbone of Stable Diffusion in a targeted, example-driven way, the pipeline learns to internalize a user-defined subject and then synthesize it within diverse contexts guided by natural language prompts. The result is a lightweight, personalized image generator that blends the flexibility of text-to-image synthesis with the precision of concept-specific learning.

### [Self-Supervised Solar Panel Detection with DINO Vision Transformers](https://github.com/amirezzati/deep-learning/blob/main/homeworks/HW5/pract/Dino.ipynb)

In this project, I applied the DINO (Self-Distillation with No Labels) model, a self-supervised Vision Transformer, to extract meaningful visual features from satellite imagery. By leveraging DINO's ability to learn representations without labeled data, I fine-tuned the model to classify images based on the presence of solar panels. Additionally, I analyzed the model's attention maps to estimate the size of solar panels in positive samples, providing insights into the model's focus areas during inference. This approach demonstrates the effectiveness of self-supervised learning in remote sensing applications, particularly in scenarios where labeled data is scarce.

### [Denoising Diffusion Probabilistic Models for CAPTCHA Image Synthesis](https://github.com/amirezzati/deep-learning/blob/main/homeworks/HW4/pract/DDPM.ipynb)

In this project, I built a full DDPM (Denoising Diffusion Probabilistic Model) pipeline from scratch to learn and generate CAPTCHA-style images. I started by defining a custom Captcha VisionDataset to load real-world CAPTCHA samples, then implemented a noise scheduler to gradually corrupt images and a UNet-inspired denoiser to predict and remove that noise. Training consisted of minimizing the mean-squared error between the model’s noise predictions and the true noise over 1,000 diffusion timesteps. After training for multiple epochs, I ran the reverse diffusion process to sample novel CAPTCHA images from pure Gaussian noise, demonstrating the model’s ability to synthesize realistic, human-readable text patterns. Generated samples were visualized alongside real data, and the final trained weights were saved for deployment or further experimentation.

### [From-Scratch SimpleGPT: A Mini Decoder-Only Transformer for Text Generation](https://github.com/amirezzati/deep-learning/blob/main/homeworks/HW3/pract/SimpleGPT.ipynb)

In this project, I implemented and trained a compact, decoder-only Transformer (“SimpleGPT”) from the ground up using PyTorch. Beginning with raw text data ingestion and vocabulary construction, I built core components—including masked self-attention heads, multi-head attention, feed-forward layers, and positional embeddings—then assembled them into a SimpleGPT architecture. After configuring training hyperparameters, I optimized the model on the dataset for autoregressive language modeling, and demonstrated its capability by generating sample text continuations. This exercise deepened my understanding of Transformer internals and highlighted how even a scaled-down GPT can learn meaningful sequence patterns.

### [Exploring Variational Autoencoders and Generative Adversarial Networks](https://github.com/amirezzati/deep-learning/blob/main/homeworks/HW4/pract/GAN-VAE.ipynb)

In this notebook, I implement and train two foundational generative modeling paradigms— a Variational Autoencoder (VAE) and a Generative Adversarial Network (GAN)—using the MNIST handwritten–digit dataset. For the VAE, I define an encoder–decoder architecture, optimize the combined reconstruction and KL‐divergence loss, and visualize both reconstructed and newly sampled digits from the learned latent space. For the GAN, I build a generator–discriminator pair, train them in an adversarial loop while tracking loss curves, and sample novel digits by feeding random noise into the generator. Finally, I compare the two models by displaying side‐by‐side sample grids and even use the trained discriminator to quantify the realism of GAN outputs, highlighting the trade-offs between probabilistic and adversarial approaches to image synthesis.

### [Defensive Distillation & Black-Box Attacks](https://github.com/amirezzati/spml/blob/main/homeworks/hw3/pract/SPML_PHW3.ipynb)

In this notebook, I first implement defensive distillation on a ResNet-based CIFAR-10 classifier by training a “teacher” network at a high softmax temperature to produce soft labels, then training an identical “student” network on those soft targets before evaluating it at temperature 1. This process strengthens the model against gradient-based adversarial perturbations. Next, I explore black-box attack methods—crafting adversarial examples without access to model gradients—by using a transfer-based attack from a surrogate model and a decision-based attack (boundary attack) to fool both the vanilla and distilled classifiers. Finally, I compare clean‐accuracy, attack success rates, and visualize representative adversarial samples to quantify how effective distillation is at mitigating black-box adversarial threats

### [Model Extraction via Knowledge Distillation](https://github.com/amirezzati/spml/blob/main/homeworks/hw4/pract/SPML_PHW4_Extraction.ipynb)

In this notebook, I first load and preprocess the CIFAR-100 dataset (resizing images to 224×224) and train a pre-trained ResNet-34 “victim” classifier on it. Then, I implement a “student” network and perform model extraction through knowledge distillation: the victim produces soft logits over batches of inputs, and the student is trained to match those logits using a KL-divergence loss. Finally, I evaluate the extracted student model against the original on clean test data, illustrating how well a distilled copy can approximate the teacher’s performance while operating under a black-box query paradigm.

### [Data Poisoning & Backdoor Attacks](https://github.com/amirezzati/spml/blob/main/homeworks/hw4/pract/SPML_PHW4_Poisoning.ipynb)

In this notebook, I first fine-tune a pre-trained ResNet18 on the CIFAR-10 dataset to establish a clean baseline. I then implement a data poisoning attack by generating adversarial “poison” examples that are mislabeled into a target class, injecting them into the training set, and retraining the network—visualizing how the poison shifts the model’s feature space. Next, I craft a watermark-based backdoor by embedding a trigger pattern into a subset of images, constructing a backdoored dataloader, and training the model to associate the trigger with an attacker’s label. Finally, I evaluate both clean and poisoned/backdoored models, demonstrating the effectiveness and stealth of these two classes of training-time attacks.


### [Comprehensive Adversarial Attacks and Robust Training on CIFAR-10 with ResNet18](https://github.com/amirezzati/spml/blob/main/homeworks/hw2/pract/Adversarial.ipynb)

In this project, I first train a ResNet18 classifier on the CIFAR-10 dataset, establishing a clean baseline. I then explore three adversarial attack methods—FGSM (via torchattacks), PGD (implemented from scratch), and UAP (Universal Adversarial Perturbations, also from scratch)—to craft perturbed examples that fool the model. After visualizing these adversarial samples, I perform adversarial training using FGSM-augmented data and retrain the network to improve its robustness. Finally, I evaluate and compare the accuracy and resilience of the original versus adversarially trained models, demonstrating how defense strategies can mitigate different types of adversarial threats.


### [An Odd Music Generator](https://github.com/amirezzati/music-generator)

This project implements an interactive audio pipeline that continuously loops through four stages to produce a unique “odd” composition. First, incoming performance audio is denoised in real time to isolate the musician’s notes from background interference. Next, a note-recognition model tracks and encodes the sequence of notes played up to each moment. Those recognized notes feed into an RNN-based prediction module that forecasts the next note to be performed. Finally, a synthetic “noise maker” injects crowd cheering and whistles back into the mix, challenging the system’s denoiser and prediction modules. By combining signal processing, sequence modeling with RNNs, and generative audio synthesis, the pipeline yields a dynamic piece of music that evolves under simulated live-performance conditions.



## Medical Image Analysis Projects
### [Graph Neural Network](https://github.com/amirezzati/iabi/blob/main/homeworks/HW4/pract/HW4_GNN.ipynb)
In this notebook, I implemented three different sections:
1. Nuclei extraction
2. Graph convolutional network     
   I implemented a GCN model to generate node embeddings of graph.       
3. Graph prediction model                     
   I implemented a GCN Graph Prediction model using the node embeddings from the GCN model and global pooling to create graph-level embeddings.

### [Interpretability](https://github.com/amirezzati/iabi/blob/main/homeworks/HW5/pract/HW5_Interpretability.ipynb)       
In this notebook, I used Grad-CAM technique as an interpretability algorithm that aids in comprehending the model's decision-making process, debugging, and explaining predictions to non-technical stakeholders. I visualized attention maps over CT-scan images to find out ResNet model focus on which features for classification.       

### [Classification using Swin Transformer](https://github.com/amirezzati/iabi/blob/main/homeworks/HW5/pract/HW5_BreastMNIST_Classification.ipynb)       
This task involves using the Swin Transformer, a cutting-edge neural network model, to distinguish between benign (including normal) and malignant cases in BreastMNIST dataset. There is a common challenge of class imbalance in this task, and I used weighted BCE loss to overcome this problem and improve the model.        
 

### [2D and 3D Segmentation](https://github.com/amirezzati/iabi/blob/main/homeworks/HW5/pract/Segmentation/HW5_Segmentation.ipynb)
In this notebook, I implemented both 2D-UNet and 3D-UNet models to perform segmentation on a set of 30 volumetric medical images.         
     

### [Medical Image Segmentation with Diffusion Model](https://github.com/amirezzati/iabi/blob/main/homeworks/HW6/pract/DDPM_MRI_Seg.ipynb)
In this notebook, I had implemented a DPM-based model for brain tumor segmentation over MRI images based on "MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model" paper.        
