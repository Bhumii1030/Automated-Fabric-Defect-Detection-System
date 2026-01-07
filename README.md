# TextileVision: Autonomous Fabric Defect Classifier
### Achieving 93.7% Accuracy with Deep Learning

## 💡 The Intuition
Quality assurance in textile manufacturing is a high-speed challenge. Detecting a 2mm tear on a fabric roll moving at 30 meters per minute is nearly impossible for the human eye over an 8-hour shift. 

My approach was to build a "Digital Inspector" that doesn't get tired. I chose **Transfer Learning** because training a model to "see" from scratch is inefficient for a 10-day sprint. By using **MobileNetV2**, I utilized a brain already expert at identifying edges and textures, allowing me to focus entirely on the specific nuances of fabric defects.

## 🛠️ The Technical Journey

### Step 1: Solving the "Noodle Image" Problem
The AITEX dataset provides images that are extremely wide ($4096 \times 256$). Feeding these directly into a standard CNN would force the model to downsample them into a blur, losing the very defects we want to find.
* **Action:** I built a custom patching script to slice these "noodles" into sixteen $256 \times 256$ square patches.
* **Impact:** This preserved the original resolution, ensuring the AI could see fine threads and small stains.

### Step 2: Balancing the Scales
In the real world, 99% of fabric is healthy. My initial dataset reflected this bias, with only ~180 defective images. An AI trained on this would simply learn to say "Healthy" every time to get a 99% score.
* **Action:** I used **Data Augmentation** (rotations, vertical/horizontal flips, and zooms) to artificially increase the defect count.
* **Result:** I achieved a balanced dataset where the AI had enough "bad examples" to learn the difference effectively.

### Step 3: Architecture & Training
I utilized a **MobileNetV2** base for its balance of speed and accuracy.
* **Freezing:** I froze the base layers to keep the pre-trained knowledge intact.
* **The Head:** I added a Global Average Pooling layer and a Dense layer with **Dropout (0.2)** to prevent the model from just memorizing the training set.
* **Optimization:** Used the **Adam optimizer** and **Binary Crossentropy** loss to fine-tune the decision-making.

## 📈 Final Performance Metrics
The model was evaluated on a "Hold-out" test set that it never saw during training to ensure the results are honest.
* **Final Test Accuracy:** 93.70%
* **Validation Accuracy:** 92.51%
* **Training Accuracy:** 95.48%

## 📂 Repository Structure
* `Fabric_Defect_Detection.ipynb`: The complete end-to-end pipeline (Data -> Training -> Prediction).
dataset link: https://www.kaggle.com/datasets/nexuswho/aitex-fabric-image-database
