
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

<img width="787" height="638" alt="image" src="https://github.com/user-attachments/assets/801759a3-0789-49b5-9a37-957c6fdc75b2" />



   
        

