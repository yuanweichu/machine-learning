# JC3509 Image Classification Comparative Study: Traditional Machine Learning vs. Deep Learning

## 1. Project Overview
This project contains the experimental code implementation for the Machine Learning course (JC3509) final assignment. The experiment compares the classification performance of a traditional statistical machine learning method (SVM + HOG features) and a modern deep learning method (CNN) on the CIFAR-10 dataset.

This experiment aims to explore and quantitatively evaluate the performance differences between "traditional feature engineering + statistical machine learning" and "modern end-to-end deep learning" in computer vision tasks. Specifically, we compared **SVM + HOG feature extraction** (Method A) against a **simple Convolutional Neural Network (CNN)** (Method B) on the CIFAR-10 dataset, and strictly conducted quantitative analyses that include both time and space complexity.

---

## 2. Environment Setup

To ensure fairness by running the comparative experiments on the same computational baseline, all models (including the CNN) in this experiment were forced to use the **CPU** for training and inference.

* **Operating System**: Windows 11
* **Python Version**: Python 3.13.0 (Clean Environment)
* **Hardware Execution**: CPU (Multi-processing parallel acceleration was enabled during the SVM phase)

### Dependencies
Please open a terminal in the project root directory and run the following command to install all necessary third-party libraries:
```bash
pip install torch torchvision scikit-learn scikit-image matplotlib numpy psutil
```

---

## 3. Dataset Configuration

* **Dataset**: CIFAR-10 (Contains 32x32 color images across 10 classes)
* **Local Storage Path**: `D:/ML_Data/cifar10`
* **Preprocessing & Sampling Strategy**: 
    To strike a balance between computational resources and statistical significance, we did not use the entire training set of 50,000 images. Instead, we applied **downsampling** by setting a random seed:
    * **Training Set**: 10,000 images
    * **Test Set**: 1,000 images

---

## 4. File Structure

```text
machine-learning/
│
├── README.md                 # Project documentation (this current file)
├── visualize.py              # Data loading and sample grid visualization script
├── train_svm.py              # Method A: Script to extract HOG features and train the SVM
├── train_cnn.py              # Method B: Script to build and train the PyTorch CNN model
├── evaluate.py               # Chart generation script (outputs CVPR-style comparison charts)
│
└── experimental_charts/ (Auto-generated)
    ├── cifar10_grid.png            # 16-image CIFAR-10 sample display
    ├── accuracy_comparison_v2.png  # Accuracy comparison bar chart (with error bars)
    ├── time_comparison_v2.png      # Training time comparison chart
    └── memory_comparison_v2.png    # Peak memory consumption comparison chart
```

---

## 5. Execution Guide

To ensure experimental reproducibility, both `train_svm.py` and `train_cnn.py` include built-in logic for **"repeating the experiment 3 times and calculating the mean and standard deviation"**.
*Note: When executing multi-processing operations on Windows, all core logic has been encapsulated within the `if __name__ == "__main__":` block to avoid recursive execution errors.*

Please execute the scripts strictly in the following order:

**Step 1: Verify Dataset and Generate Sample Grid**
```bash
python visualize.py
```
*Expected Output: A matplotlib window will pop up displaying 16 CIFAR-10 sample images.*

**Step 2: Run the Baseline Model (SVM + HOG)**
```bash
python train_svm.py
```
*Expected Output: Automatically initiates multi-processing to extract HOG features for 10,000 images. After 3 loops, it prints the average accuracy, time overhead, and memory peak in the terminal. Estimated time: 1-2 minutes.*

**Step 3: Run the Deep Learning Model (CNN)**
```bash
python train_cnn.py
```
*Expected Output: Builds a basic 3-layer convolutional network, runs 3 complete 5-Epoch training loops, and prints the average metrics. Estimated time: ~1 minute.*

**Step 4: Generate Paper Figures**
```bash
python evaluate.py
```
*Expected Output: Generates three high-resolution `.png` charts in the current directory, ready to be directly inserted into the Quantitative Evaluation section of the assignment report.*

---

## 6. Summary of Findings

This experiment is based on 10,000 training samples, repeated 3 times to obtain average results.

| Metrics | Traditional Method (SVM + HOG) | Deep Learning Method (CNN) | Conclusion / Trade-off |
| :--- | :--- | :--- | :--- |
| **Test Set Accuracy** | 55.13% ± 0.54% | **58.40% ± 0.99%** | CNN's end-to-end feature learning capability outperforms manually designed HOG features. |
| **Average Total Time** | 17.37 seconds | **15.79 seconds** | After scaling up the data volume, the traditional SVM's time complexity grows exponentially due to feature extraction and kernel function calculations, making it slower than the CNN. |
| **Peak Memory Consumption**| **71.57 MB** | 72.97 MB | The CNN contains approximately 620,000 parameters, resulting in slightly higher space complexity. |

**Academic Insight**: The experiment demonstrates that under the constraints of extremely low resolution (32x32) and a larger sample size (10,000), the CNN outperforms the traditional statistical learning method in both recognition accuracy and time scalability.

---

