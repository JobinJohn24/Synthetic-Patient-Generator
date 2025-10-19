# Synthetic-Patient-Generator

## 📌 Project Overview
This project uses a Generative Adversarial Network (GAN) to create realistic synthetic patient data for use in pharmaceutical and healthcare research. The goal is to support meaningful analysis while keeping patient information private and secure. The model is trained on anonymized patient records, learning the patterns and statistics of real data so it can generate new, lifelike datasets that reflect the same trends—without revealing anyone’s personal details.


## 🔬 Workflow
1. Data Preparation: Load the anonymized patient dataset in .npy format.

2. Model Training: Train a Wasserstein GAN to improve stability during learning.

3. Synthetic Generation: Use the trained model to create realistic synthetic patient samples.

4. Validation: Check that privacy is protected and the synthetic data matches real-world patterns.

5. Deployment: Release the validated synthetic dataset for research purposes.

---

## ⚙️ Technical Methods & Models
- **Language:** Python  
- **Algorithms:** GANs, Variational Autoencoders  
- **Libraries:** TensorFlow, PyTorch  

---

## 📊 Visualizations
* Real vs synthetic feature distributions.

[realvfakegraph](

- PCA/UMAP projection of data space.  
- Privacy leakage metrics.  

---

## 📚 Learning Resources
- *Deep Learning (Goodfellow et al.)* – ISBN: 9780262035611  
- *Practical Synthetic Data Generation* – ISBN: 9781801812639  
- YouTube: [Two Minute Papers – GANs](https://www.youtube.com/@TwoMinutePapers)  

---

