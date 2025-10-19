# Synthetic Patient Generator

## 📌 Project Overview
This project uses a Generative Adversarial Network (GAN) to create realistic synthetic patient data for use in pharmaceutical and healthcare research. The goal is to support meaningful analysis while keeping patient information private and secure. The model is trained on anonymized patient records, learning the patterns and statistics of real data so it can generate new, lifelike datasets that reflect the same trends—without revealing anyone’s personal details. Real patient datasets are locked behind privacy laws, IRB approval, and data sharing agreements

## 🔬 Workflow
1. Data Preparation: Load the anonymized patient dataset in .npy format.

2. Model Training: Train a Wasserstein GAN to improve stability during learning.

3. Synthetic Generation: Use the trained model to create realistic synthetic patient samples.

4. Validation: Check that privacy is protected and the synthetic data matches real-world patterns.

5. Deployment: Release the validated synthetic dataset for research purposes.

## 📈Data Interpretation
1. Generates 500+ synthetic patient records from 297 real samples

2. Privacy risk under 0.5% (minimal re-identification threat)

3. Uses Wasserstein GAN for stable, reliable training

## ⚙️ Technical Methods & Models
- **Language:** Python  
- **Algorithms:** GANs, Variational Autoencoders  
- **Libraries:** TensorFlow, PyTorch

## 👨‍💻 Installation

git clone https://github.com/JoinJohn24/Synthetic-Patient-Generator.git

cd Synthetic-Patient-Generator

pip install -r requirements.txt

## 📊 Visualizations
* K-means clustering showed how synthetic samples are distrbuted across clusters

PCA/UMAP projection of data space.


|Metric|Value|
|-|-|
|Synthetic Samples generated|500|
|Privacy Risk|0.34%|
|Real Cluster Tightness|2.91 average|
|Synthetic Cluster Spread|5.44|
|PCA Variance Captured|36% in 2D|
