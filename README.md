
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

Workflow

Dataset Collection & Annotation

1074+ IR images collected from Roboflow.

Annotated with bounding boxes for hot spot regions.

Data Augmentation

Zoom, rotation, brightness/contrast, translation, noise injection, and cutout applied.

Model Fine-Tuning

Faster R-CNN ResNet-50 trained with custom dataset.

Evaluation

Precision, Recall, mAP, and loss monitored using TensorBoard.

Inference

Model tested on unseen real-world IR images from UNITEN campus and Roboflow.

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

⚙️ Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/yourusername/PV-HotSpot-Detection.git
cd PV-HotSpot-Detection

2️⃣ Install Dependencies
pip install -r requirements.txt


Required libraries include:

tensorflow==2.x
tensorflow-gpu==2.x
opencv-python
pandas
matplotlib
lxml
protobuf
Pillow

3️⃣ Download TensorFlow Model Zoo

Download Faster R-CNN ResNet-50 from TensorFlow 2 Model Zoo:
👉 https://github.com/tensorflow/models

Extract the checkpoint into:

/models/faster_rcnn_resnet50/

4️⃣ Prepare the Dataset

Place training images in images/train/ and test images in images/test/.

Run the annotation script to generate TFRecord files:

python scripts/generate_tfrecord.py --csv_input=annotations/train_labels.csv --output_path=annotations/train.record
python scripts/generate_tfrecord.py --csv_input=annotations/test_labels.csv --output_path=annotations/test.record

5️⃣ Update pipeline.config

Edit:

num_classes (set to 1 for “hot spot”)

fine_tune_checkpoint path

train_input_reader and eval_input_reader TFRecord paths

Batch size, learning rate, and number of steps as desired.

🚀 Training the Model

Run the following command in your Colab or terminal:

python model_main_tf2.py \
  --model_dir=models/faster_rcnn_resnet50 \
  --pipeline_config_path=pipeline.config \
  --num_train_steps=62000


Monitor training progress in TensorBoard:

tensorboard --logdir=models/faster_rcnn_resnet50

🧪 Evaluation

Evaluate the model after training:

python model_main_tf2.py \
  --model_dir=models/faster_rcnn_resnet50 \
  --pipeline_config_path=pipeline.config \
  --checkpoint_dir=models/faster_rcnn_resnet50

🔍 Exporting the Trained Model
python exporter_main_v2.py \
  --input_type image_tensor \
  --pipeline_config_path pipeline.config \
  --trained_checkpoint_dir=models/faster_rcnn_resnet50 \
  --output_directory=exported-models/my_model

🖼️ Running Inference

To detect hot spots on new IR images:

from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf
import cv2

# Load model
detect_fn = tf.saved_model.load("exported-models/my_model/saved_model")

# Load image
image_path = "test_image.jpg"
image_np = cv2.imread(image_path)
input_tensor = tf.convert_to_tensor([image_np])
detections = detect_fn(input_tensor)

# Visualize
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np,
    detections['detection_boxes'][0].numpy(),
    detections['detection_classes'][0].numpy().astype(int),
    detections['detection_scores'][0].numpy(),
    {1: {'id': 1, 'name': 'hot spot'}},
    use_normalized_coordinates=True,
    line_thickness=3,
)

cv2.imshow('PV Hot Spot Detection', image_np)
cv2.waitKey(0)

📈 Results Visualization

Example inference results:

Example	Hot Spot Type	Detection
Figure 4.4.1	Far-Distant Hot Spot	✅ Detected
Figure 4.4.2	Square Large Hot Spot	✅ High Confidence
Figure 4.4.3	Medium Hot Spot	⚠️ Moderate Confidence
🧾 Citation

If you use this project, please cite as:

A. M. A. Abdallaty and P. S. Krishnan, “Development of a Photovoltaic Hot Spot Detection System Using Artificial Intelligence,” 2025.

🤝 Contributing

Feel free to open issues or pull requests to improve:

Dataset diversity

Hyperparameter tuning

Model speed and inference optimization

💡 Future Work

Integrating IoT for real-time PV inspection

Deploying the model on edge devices (e.g., Jetson Nano)

Improving detection for complex and dimmed hot spots
