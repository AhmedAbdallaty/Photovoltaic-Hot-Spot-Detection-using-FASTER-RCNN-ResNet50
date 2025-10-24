
🌞 Development of a Photovoltaic Hot Spot Detection System Using Artificial Intelligence

🧠 Overview

This project presents the development of an AI-based Photovoltaic (PV) Hot Spot Detection System using the Faster R-CNN with ResNet-50 backbone. The goal is to detect PV hot spots in infrared (IR) thermographic images, which indicate localized overheating and potential panel damage.

Unlike earlier models such as YOLOv5, SSD, or Faster R-CNN VGG16, this work enhances detection performance by training on a diverse and augmented dataset containing PV hot spots of various shapes, sizes, distances, and thermal palettes, improving generalization and robustness in real-world inspection scenarios.

📊 Model Summary

F1 Score: 65.19%
Precision: (50% IoU)	80.06%
Recall: (100 Detections)	54.89%
Best Detection:	Large & simple hot spots
Weakest Detection:	Complex (dimmed, linear, circular) hot spots

The model was trained for 62,000 steps using TensorFlow 2 Object Detection API on Google Colab with GPU/TPU acceleration.

🏗️ System Architecture

Model: Faster R-CNN with ResNet-50 backbone
Framework: TensorFlow 2 Object Detection API
Training Environment: Google Colab
Dataset Source: Roboflow (annotated and augmented IR PV images)
Annotation Format: TFRecord and label map

🧩 Directory Structure

📦 PV-HotSpot-Detection
 ┣ 📂 annotations/
 ┃ ┣ train.record
 ┃ ┣ test.record
 ┃ ┗ label_map.pbtxt
 ┣ 📂 images/
 ┃ ┣ train/
 ┃ ┗ test/
 ┣ 📂 models/
 ┃ ┗ faster_rcnn_resnet50/
 ┣ 📂 exported-models/
 ┃ ┗ my_model/
 ┣ 📂 scripts/
 ┃ ┗ generate_tfrecord.py
 ┣ 📜 pipeline.config
 ┣ 📜 training_script.ipynb
 ┣ 📜 requirements.txt
 ┗ 📜 README.md


🚀 Fine-Tuning Your Model

1. After data preprocessing, make sure your Google Drive repository is structured as follows:
        <img width="411" height="337" alt="image" src="https://github.com/user-attachments/assets/1d2ff8c3-56a1-40de-897e-78a3d407c29f" />
2. Make sure you have already saved the training and validation label map and TFRecord files in the annotations folder.
3. Mount your Google Drive:
   `from google.colab import drive
   drive.mount('/content/drive')`
4. Install latest TensorFlow version:
   `!pip install tensorflow==2.13`
5. Clone TensorFlow model:
   `!git clone https://github.com/tensorflow/models.git`
6. `pwd `/conten``
   `cd /content/models/research`
   '!protoc object_detection/protos/*.proto --python_out=.'
   
        

