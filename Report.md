# Technical Report: Automated Fabric Defect Detection
**Objective:** To develop a high-accuracy binary classifier for identifying textile defects using Deep Learning.

---

## 1. Problem Statement
Manual inspection in textile manufacturing is subject to human fatigue and inconsistency. The industrial challenge lies in the high-resolution nature of the data and the rarity of defects compared to healthy fabric (Class Imbalance). This project aims to automate this via a Convolutional Neural Network (CNN).

## 2. Dataset & Preprocessing
The **AITEX Fabric Image Database** was used, consisting of high-resolution "strip" images ($4096 \times 256$ pixels).

### 2.1 The Patching Strategy
Standard CNN architectures cannot ingest $4096 \times 256$ images without extreme downsampling, which destroys the fine-grained features of defects (stains, holes, snags). 
* **Method:** I implemented a sliding-window patching algorithm.
* **Result:** Sliced the original images into sixteen $256 \times 256$ squares.
* **Intuition:** Preserving the original pixel density ensures the model can detect minute structural anomalies.



### 2.2 Addressing Class Imbalance
The initial patching resulted in 1,464 Healthy patches but only 183 Defective patches (an 8:1 ratio).
* **Strategy:** Synthetic Data Augmentation.
* **Technique:** Performed random rotations (40°), horizontal/vertical flips, and zooms on the minority class.
* **Final Set:** Balanced the dataset to 2,681 total samples (1,217 Defect, 1,464 Healthy).



## 3. Model Architecture
I utilized **Transfer Learning** to leverage pre-existing feature extraction knowledge.

* **Base Model:** MobileNetV2 (Pre-trained on ImageNet).
* **Reasoning:** MobileNetV2 is optimized for mobile/edge devices, making it ideal for future integration into factory floor cameras.
* **Custom Layers:**
    * **Global Average Pooling:** Reduced the spatial dimensions.
    * **Dropout (0.2):** Implemented to prevent overfitting on the limited dataset.
    * **Sigmoid Output:** Used for binary classification (Defect vs. No Defect).

## 4. Experimental Results
The model was trained for 10 epochs using the **Adam Optimizer** and **Binary Crossentropy Loss**.

### 4.1 Performance Metrics
* **Final Test Accuracy:** 93.70%
* **Validation Accuracy:** 92.51%
* **Training Accuracy:** 95.48%

### 4.2 Observations
The proximity of training and validation accuracy indicates that the model is generalizing well. The "Loss" curves stabilized after Epoch 6, suggesting the model reached convergence efficiently.



## 5. Conclusion & Future Work
The project successfully demonstrates that a lightweight CNN (MobileNetV2) can identify fabric defects with over 93% accuracy. 

**Future Steps:**
1. **Object Detection:** Transition from classification to YOLOv8 to locate the exact coordinates of defects.
2. **Multi-class Classification:** Expand the model to distinguish between types of defects (e.g., oil stains vs. holes).
