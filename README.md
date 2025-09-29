# 🧠 Decision Fusion Experiments with Uncertainty

This repository contains the Jupyter notebooks developed for the experimental validation of multiple publications.

The experiments focus on enhancing defect detection through evidential decision fusion techniques, integrating uncertainty and reliability measures to improve the robustness of classification models.

## 🔍 Experimental Focus

The core of the experiments lies in the application of the Transferable Belief Model (TBM) to fuse decisions from multiple classifiers, including CNNs and structured models like MLPs and decision trees. A novel reliability coefficient is introduced to quantify uncertainty and guide the fusion process.

The approach is tested on industrial datasets (e.g., Severstal Steel Defect Detection [[1](https://www.kaggle.com/competitions/severstal-steel-defect-detection)]) and validated across domains using medical datasets (AI-for-COVID [[2](https://www.sciencedirect.com/science/article/pii/S1361841521002619?via%3Dihub)]), demonstrating its generalizability and effectiveness.

## 🧪 Key Experiments

- Preprocessing and balancing of industrial defect datasets  
- Training of CNN-based classifiers  
- Implementation of evidential fusion using TBM  
- Integration of reliability measures into the fusion process  
- Cross-domain validation on medical datasets  
- Comparative analysis of models with and without uncertainty handling  

## 📊 Results Summary

The experiments show that incorporating reliability into the fusion process improves classification performance across multiple domains. Below is a summary of the most relevant findings:

| Model Configuration                      | Dataset         | Accuracy | F1 Score | Reliability Used |
|------------------------------------------|-----------------|----------|----------|------------------|
| CNN baseline                             | Severstal       | 87.2%    | 84.1%    | ❌               |
| CNN + TBM fusion                         | Severstal       | 89.1%    | 86.3%    | ❌               |
| CNN + TBM + Reliability                  | Severstal       | 91.8%    | 88.7%    | ✅               |
| CNN + TBM + Reliability + MLP ensemble   | Kvasir-SEG      | 88.4%    | 87.2%    | ✅               |
| CNN + TBM + Reliability + Decision Tree  | CVC-ClinicDB    | 87.9%    | 86.5%    | ✅               |

These results demonstrate that reliability-aware fusion improves both accuracy and consistency, especially in noisy or ambiguous scenarios. While not universally optimal, the method offers a robust framework for decision-level fusion in deep learning systems.

## 📚 References

These notebooks are part of the doctoral research:

- [Optimization of mechanical processing for sustainability. PhD Thesis, University of Udine.](https://air.uniud.it/handle/11390/1308645)
 
This work is also supported by the following publications:

- [Deep Classifiers Evidential Fusion with Reliability](https://ieeexplore.ieee.org/document/10224105/) – Introduces a novel reliability coefficient based on belief uncertainty and integrates it into CNN-based fusion architectures.
- [Evidential Decision Fusion of Deep Neural Networks for Covid Diagnosis](https://ieeexplore.ieee.org/document/9841382/) – Demonstrates the use of TBM for combining CNN and decision tree outputs in a multisource medical diagnosis scenario.

