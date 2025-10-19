# Synthetic Patient Generator

## 📌 Project Overview
This project uses a Generative Adversarial Network (GAN) to create realistic synthetic patient data for use in pharmaceutical and healthcare research. The goal is to support meaningful analysis while keeping patient information private and secure. The model is trained on anonymized patient records, learning the patterns and statistics of real data so it can generate new, lifelike datasets that reflect the same trends—without revealing anyone’s personal details. Real patient datasets are locked behind privacy laws, IRB approval, and data sharing agreements

## 📋 Implementation 
This project can be a stepping stone in creating tools that bridge the gap between real-world healthcare data and safe, privacy-preserving research. The synthetic patient generator can be used in a multitude of scenarios:
  
  * Medical Research - Researchers can test hyptheses, explore disease models, and simulate treatment outcomes without needing immediate access to sensitive patient data.
  
  * Algorithm Development - Data Scientists can build, train, and implement the stress-test machine learning models on synthetic cohorts before applying them to clinical datasets (limited or regulated)

  * Policy & Planning Simulations - Public health analysts can use this to model population-level and study potential impacts before rollout.

  * Healthcare Software - Clinical decision support tools or EHR systems populating their applications with synthetic patients to test features, workflows, and analytics without the privacy concerns.

## 🔬 Workflow
1. Data Preparation: Load the anonymized patient dataset in `.npy` format.

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
- **Libraries:** TensorFlow, Numpy, Scikit-learn, Matplotlib

## 👨‍💻 Installation


git clone `https://github.com/JobinJohn24/Synthetic-Patient-Generator.git`

cd `Synthetic-Patient-Generator`

pip install -r `requirements.txt`

## 👨‍🔬 Results

After training 297 real patients records: 
|Metric|Value|
|-|-|
|Synthetic Samples generated|500|
|Privacy Risk|0.34%|
|Real Cluster Tightness|2.91 average|
|Synthetic Cluster Spread|5.44|
|PCA Variance Captured|36% in 2D|

## 📊 Visualizations
* K-means clustering showed how synthetic samples are distributed across clusters.
![kmeans](https://github.com/JobinJohn24/Synthetic-Patient-Generator/blob/main/images/kmeans_statistics.png)


* PCA Projection & Feature Distribution shows the overlap in data between the synthetic and real datasets based on individual features
![pcaprojection](https://github.com/JobinJohn24/Synthetic-Patient-Generator/blob/main/images/pca_projection.png)
![feature distribution](https://github.com/JobinJohn24/Synthetic-Patient-Generator/blob/main/images/validation_results.png)
