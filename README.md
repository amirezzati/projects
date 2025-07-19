# Projects


### [Self-Supervised Solar Panel Detection with DINO Vision Transformers](https://github.com/amirezzati/deep-learning/blob/main/homeworks/HW5/pract/Dino.ipynb)

In this project, I applied the DINO (Self-Distillation with No Labels) model, a self-supervised Vision Transformer, to extract meaningful visual features from satellite imagery. By leveraging DINO's ability to learn representations without labeled data, I fine-tuned the model to classify images based on the presence of solar panels. Additionally, I analyzed the model's attention maps to estimate the size of solar panels in positive samples, providing insights into the model's focus areas during inference. This approach demonstrates the effectiveness of self-supervised learning in remote sensing applications, particularly in scenarios where labeled data is scarce.

### [Adversarial Attacks and Robust Training on CIFAR-10 with ResNet18](https://github.com/amirezzati/deep-learning/blob/main/homeworks/HW5/pract/Adversarial_attacks_training.ipynb)

In this project, I trained a ResNet18 classifier on the CIFAR-10 dataset and investigated its vulnerability to adversarial perturbations. I implemented two attack methods—FGSM (using a library) and PGD (from scratch)—to generate adversarial examples, then used FGSM to perform adversarial training. Finally, I evaluated and compared the accuracy and robustness of the original versus adversarially trained models, highlighting the impact of these defenses on model performance.


### [Fine-Tuning of Stable Diffusion for Custom Concept Generation](https://github.com/amirezzati/deep-learning/blob/main/homeworks/HW5/pract/StableDiffusion.ipynb)

This project demonstrates how to adapt a large-scale diffusion model to capture and reproduce a novel visual concept from just a handful of examples. By fine-tuning only the core generative backbone of Stable Diffusion in a targeted, example-driven way, the pipeline learns to internalize a user-defined subject and then synthesize it within diverse contexts guided by natural language prompts. The result is a lightweight, personalized image generator that blends the flexibility of text-to-image synthesis with the precision of concept-specific learning.


### [Denoising Diffusion Probabilistic Models for CAPTCHA Image Synthesis](https://github.com/amirezzati/deep-learning/blob/main/homeworks/HW4/pract/DDPM.ipynb)

In this project, I built a full DDPM (Denoising Diffusion Probabilistic Model) pipeline from scratch to learn and generate CAPTCHA-style images. I started by defining a custom Captcha VisionDataset to load real-world CAPTCHA samples, then implemented a noise scheduler to gradually corrupt images and a UNet-inspired denoiser to predict and remove that noise. Training consisted of minimizing the mean-squared error between the model’s noise predictions and the true noise over 1,000 diffusion timesteps. After training for multiple epochs, I ran the reverse diffusion process to sample novel CAPTCHA images from pure Gaussian noise, demonstrating the model’s ability to synthesize realistic, human-readable text patterns. Generated samples were visualized alongside real data, and the final trained weights were saved for deployment or further experimentation.


### [Exploring Variational Autoencoders and Generative Adversarial Networks on MNIST](https://github.com/amirezzati/deep-learning/blob/main/homeworks/HW4/pract/GAN-VAE.ipynb)

In this notebook, I implement and train two foundational generative modeling paradigms— a Variational Autoencoder (VAE) and a Generative Adversarial Network (GAN)—using the MNIST handwritten–digit dataset. For the VAE, I define an encoder–decoder architecture, optimize the combined reconstruction and KL‐divergence loss, and visualize both reconstructed and newly sampled digits from the learned latent space. For the GAN, I build a generator–discriminator pair, train them in an adversarial loop while tracking loss curves, and sample novel digits by feeding random noise into the generator. Finally, I compare the two models by displaying side‐by‐side sample grids and even use the trained discriminator to quantify the realism of GAN outputs, highlighting the trade-offs between probabilistic and adversarial approaches to image synthesis.


### [From-Scratch SimpleGPT: A Mini Decoder-Only Transformer for Text Generation](https://github.com/amirezzati/deep-learning/blob/main/homeworks/HW3/pract/SimpleGPT.ipynb)

In this project, I implemented and trained a compact, decoder-only Transformer (“SimpleGPT”) from the ground up using PyTorch. Beginning with raw text data ingestion and vocabulary construction, I built core components—including masked self-attention heads, multi-head attention, feed-forward layers, and positional embeddings—then assembled them into a SimpleGPT architecture. After configuring training hyperparameters, I optimized the model on the dataset for autoregressive language modeling, and demonstrated its capability by generating sample text continuations. This exercise deepened my understanding of Transformer internals and highlighted how even a scaled-down GPT can learn meaningful sequence patterns.


