# RAI-FinalProject
# Interpretable Image Classification via Deep Supervised Clustering
Marius Arvinte, Mai Lee Chang and (Ethan) Yuqiang Heng

Responsible AI, UT Austin, Spring 2019

## Instructions
1. Run 'main_adversarial.py' with 'op_mode = blank' to train a number of adversarial autoencoders for a given dataset. All autoencoders will be saved and can be used later.

2. Open 'silhouettes.txt' in the result folder and pick the best autoencoder (highest silhouette score). Write down its global seed (same for all), weight seed, try number and epoch number for which the score is highest.

3. Run 'main_adversarial.py' with 'op_mode = preloaded' to load a pretrained autoencoder by its seeding parameters and to generate latent representation of a given dataset. The latent representation will be saved as a .mat file and can be used later.

4. Run 'main_ensemble_classifier.py' to train an ensemble of 1-D classifier based on pregenerated latent representations. This script can be configured to work with 'embedding_type = {euclidean, cosine, l1}' and adds prototype vectors according to the proposed max-variance-first criteria in the paper.

5. (Optional) Run 'main_synthetic_data_classifier.py' to generate synthetic data and train an incrementally increasing joint classifier with random or proposed ordering (for multiple distance embeddings).
