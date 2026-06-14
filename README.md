# Date Fruit Classification using ResNet-50 and Custom CNN

![Thumbnail](https://raw.githubusercontent.com/saqibbedar/date-fruit-classification-resnet50-and-customCNN/refs/heads/main/assets/thumbnailv2.jpg)

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/saqibbedar/date-fruit-classification-resnet50-and-customCNN)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/saqibbedar/date-fruit-classification-resnet50)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)
![Status](https://img.shields.io/badge/Status-Academic%20Project-brightgreen)

A computer vision project for classifying date fruit varieties using **ResNet-50 transfer learning** and a **custom CNN model trained from scratch** with **PyTorch**.

The project includes dataset preparation, model training, fine-tuning, custom model comparison, evaluation, confusion matrix analysis, and a simple **Flask web application** for image upload and prediction.

## Project Overview

This project classifies date fruit images into 10 varieties:

- Ajwa
- Aseel
- Fasli-Toto
- Galaxy
- Medjool
- Meneifi
- Nabtat-Ali
- Shaishe
- Sokari
- Sugaey

The main model uses **ResNet-50 transfer learning** with ImageNet pretrained weights. The original ResNet-50 classifier was replaced with a new 10-class classifier for date fruit classification.

A second model, **CustomDateCNN**, was also trained from scratch without pretrained weights. This model was added as a comparison baseline to understand the difference between transfer learning and a simple custom convolutional neural network.

## Project Objective

The objective of this project is to develop a computer vision-based date fruit classification system that can identify different date varieties from images.

The project demonstrates a complete machine learning workflow, including dataset preparation, model training, evaluation, comparison between a pretrained ResNet-50 model and a custom CNN model, and deployment through a simple web application.

This work shows how deep learning can support agricultural image classification tasks such as fruit sorting, labeling, variety identification, and quality inspection.

## Features

- ResNet-50 transfer learning
- Frozen-backbone baseline training
- Fine-tuning of ResNet `layer4`
- Custom CNN model trained from scratch
- Comparison between pretrained and non-pretrained models
- Train/validation/test split
- Image preprocessing and augmentation
- Confusion matrix evaluation
- Precision, recall, and F1-score analysis
- Flask web app for image upload and prediction
- Confidence thresholding
- Basic date-like image precheck for rejecting obvious non-date inputs
- Kaggle dataset and project package

## Dataset

The dataset contains 7,000 images distributed equally across 10 classes.

| Split      | Images |
| ---------- | -----: |
| Training   |  4,890 |
| Validation |  1,050 |
| Test       |  1,060 |
| Total      |  7,000 |

Each class contains 700 images.

Dataset split:

- 70% training
- 15% validation
- 15% testing

## Model 1: ResNet-50 Transfer Learning

Architecture:

- Backbone: ResNet-50
- Pretrained weights: ImageNet
- Transfer learning: Yes
- Input size: 224 × 224 RGB image
- Original classifier: `Linear(2048, 1000)`
- Modified classifier head:

```python
Dropout(p=0.30)
Linear(2048, 10)
```

Training strategy:

1. Freeze ResNet-50 backbone and train only the classifier head.
2. Save the best frozen-backbone model.
3. Load the best frozen model.
4. Unfreeze ResNet `layer4` and classifier head.
5. Fine-tune with smaller learning rates.

### ResNet-50 Metadata

| Item                         | Value                                    |
| ---------------------------- | ---------------------------------------- |
| Model name                   | ResNet-50 Fine-Tuned Model               |
| Framework                    | PyTorch / Torchvision                    |
| Pretrained                   | Yes                                      |
| Pretrained dataset           | ImageNet                                 |
| Transfer learning            | Yes                                      |
| Number of classes            | 10                                       |
| Input size                   | 224 × 224                                |
| Classifier head              | Dropout + Linear(2048 → 10)              |
| Loss function                | CrossEntropyLoss                         |
| Optimizer                    | AdamW                                    |
| Weight decay                 | 0.0001                                   |
| Scheduler                    | ReduceLROnPlateau                        |
| Stage 1 trainable parameters | 20,490                                   |
| Stage 2 trainable parameters | 14,985,226                               |
| Total parameters             | 23,528,522                               |
| Final checkpoint             | `checkpoints/best_resnet50_finetuned.pt` |

## Model 2: CustomDateCNN From Scratch

The second model is a simple custom CNN trained from scratch. It does not use ImageNet, pretrained weights, or transfer learning.

This model was created to compare a lightweight custom architecture with the ResNet-50 transfer learning approach.

Architecture:

```text
Input Image: 3 × 128 × 128

Block 1:
Conv2d(3 → 32), BatchNorm, ReLU, MaxPool

Block 2:
Conv2d(32 → 64), BatchNorm, ReLU, MaxPool

Block 3:
Conv2d(64 → 128), BatchNorm, ReLU, MaxPool

Block 4:
Conv2d(128 → 256), BatchNorm, ReLU, MaxPool

Classifier:
AdaptiveAvgPool2d(1 × 1)
Flatten
Dropout(0.30)
Linear(256 → 10)
```

The channel depth increases as:

```text
3 → 32 → 64 → 128 → 256
```

The spatial size decreases as:

```text
128 × 128 → 64 × 64 → 32 × 32 → 16 × 16 → 8 × 8
```

This means the model gradually reduces image size while increasing feature depth, allowing it to learn stronger visual patterns such as color, shape, shine, and wrinkle texture.

### CustomDateCNN Metadata

| Item                     | Value                                        |
| ------------------------ | -------------------------------------------- |
| Model name               | CustomDateCNN                                |
| Framework                | PyTorch                                      |
| Pretrained               | No                                           |
| Transfer learning        | No                                           |
| Training type            | From scratch                                 |
| Number of classes        | 10                                           |
| Input size               | 128 × 128                                    |
| Main filter size         | 3 × 3                                        |
| Convolution blocks       | 4                                            |
| Channel flow             | 3 → 32 → 64 → 128 → 256                      |
| Classifier               | AdaptiveAvgPool + Flatten + Dropout + Linear |
| Final output layer       | Linear(256 → 10)                             |
| Loss function            | CrossEntropyLoss                             |
| Optimizer                | AdamW                                        |
| Learning rate            | 0.001                                        |
| Weight decay             | 0.0001                                       |
| Scheduler                | ReduceLROnPlateau                            |
| Total parameters         | 391,946                                      |
| Trainable parameters     | 391,946                                      |
| Best validation accuracy | 96.38%                                       |
| Best epoch               | 14                                           |
| Final checkpoint         | `checkpoints/best_custom_cnn.pt`             |

## Results

### ResNet-50 Frozen-Backbone Training

Best validation accuracy:

```text
88.86%
```

### ResNet-50 Fine-Tuned Model

Best validation accuracy:

```text
94.86%
```

Final test accuracy:

```text
93.96%
```

Final test loss:

```text
0.1730
```

### CustomDateCNN From Scratch

Best validation accuracy:

```text
96.38%
```

Best epoch:

```text
14
```

The custom CNN was trained as a baseline comparison model. Although it performed strongly on the validation set, the ResNet-50 model remains important because it uses a deeper pretrained backbone and is expected to generalize better in more diverse real-world conditions.

## Model Comparison

| Feature                  | ResNet-50          | CustomDateCNN        |
| ------------------------ | ------------------ | -------------------- |
| Pretrained               | Yes                | No                   |
| Transfer learning        | Yes                | No                   |
| Training type            | Fine-tuning        | From scratch         |
| Input size               | 224 × 224          | 128 × 128            |
| Architecture depth       | Deep residual CNN  | Simple CNN           |
| Main feature extractor   | ResNet-50 backbone | 4 custom conv blocks |
| Final classifier         | Linear(2048 → 10)  | Linear(256 → 10)     |
| Total parameters         | 23,528,522         | 391,946              |
| Best validation accuracy | 94.86%             | 96.38%               |
| Purpose                  | Main model         | Comparison baseline  |

## Kaggle Dataset + Project Package

The public Kaggle package is available here:

```text
https://www.kaggle.com/datasets/saqibbedar/date-fruit-classification-resnet50
```

The Kaggle package includes the dataset, trained checkpoints, source code, split manifest files, and project documentation.

## Confusion Matrix Observation

The ResNet-50 model performs very well on most classes such as Ajwa, Medjool, Shaishe, Aseel, and Fasli-Toto.

The most challenging class is **Meneifi**, which is sometimes confused with:

- Medjool
- Sugaey
- Galaxy

This shows that some date varieties are visually similar and require deeper error analysis, improved data quality, or additional real-world samples.

## Project Structure

```text
date-fruit-classification-resnet50-and-customcnn/
│
├── assets/
│   ├── thumbnailv2.jpg
│   └── resnet_architecture.webp
│
├── checkpoints/
│   ├── best_resnet50_frozen.pt
│   ├── best_resnet50_finetuned.pt
│   └── best_custom_cnn.pt
│
├── datasets/
│   ├── full/
│   ├── train/
│   ├── val/
│   └── test/
│
├── kaggle/
│   ├── DATASET_CARD.md
│   └── dataset-metadata.template.json
│
├── reports/
│   ├── confusion_matrix_test.png
│   ├── classification_metrics_test.csv
│   ├── custom_cnn_confusion_matrix_test.png
│   └── custom_cnn_metrics_test.csv
│
├── scripts/
│   └── create_kaggle_split_manifest.py
│
├── static/
│   └── uploads/              # generated by Flask app
│
├── .gitattributes
├── .gitignore
├── app.py
├── conda_config_cmds.txt
├── custom_model.ipynb
├── index.html
├── main.ipynb
├── main.py
├── pyproject.toml
├── requirements-demo-cpu.txt
├── requirements-dev.txt
└── README.md
```

## Installation and Configuration

The project can be configured in different ways depending on the system. A simple CPU setup is enough for running the Flask demo, while GPU setup is recommended for training and retraining the models.

### Clone Repository

Clone the repository and open it in your desired IDE.

```bash
git clone https://github.com/saqibbedar/date-fruit-classification-resnet50-and-customCNN.git
```

### Option 1: Python `venv` Setup for CPU Demo

Use this option on a laptop or system without NVIDIA GPU. This setup is suitable for running the trained model through the Flask web application.

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment on Windows:

```bash
.\.venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install CPU version of PyTorch:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Install required packages:

```bash
python -m pip install flask numpy pillow matplotlib ipykernel
```

Run the Flask app:

```bash
python app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

### Option 2: Miniconda Setup for Local Training and GPU Work

Miniconda is recommended for local development because it is lightweight compared to the full Anaconda distribution. It is a good option for GPU-based PyTorch training, Jupyter Notebook work, and managing separate project environments.

Create a new conda environment:

```bash
conda create -n datefruit-cv python=3.12 -y
```

Activate the environment:

```bash
conda activate datefruit-cv
```

Install CUDA-supported PyTorch:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install remaining project dependencies:

```bash
python -m pip install flask numpy pillow matplotlib ipykernel
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name datefruit-cv --display-name "Python (datefruit-cv)"
```

Run the Flask app:

```bash
python app.py
```

For additional conda setup, activation, package checking, and GPU verification commands, see:

```text
conda_config_cmds.txt
```

### Option 3: Full Anaconda Setup

If Miniconda is not installed, the full Anaconda distribution can also be used. Download Anaconda from the official Anaconda website, install it, open Anaconda Prompt, and then follow the same conda environment commands from Option 2.

This option is easier for beginners because it includes many scientific Python tools by default, but it is larger in size than Miniconda.

### GPU Verification

To confirm whether PyTorch can access the GPU, run:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If the output shows `True`, CUDA is available and the model can use the GPU. If it shows `False`, the project can still run on CPU for inference and demo purposes.

## Running the Web App

Start the app:

```bash
python app.py
```

Then upload an image using the browser interface.

The app shows:

- predicted class
- confidence score
- top-k predictions
- accepted/rejected decision
- simple input quality check

## Important Note about Unknown Images

This is a closed-set classifier.

That means the model was trained only on 10 known date classes. If a non-date image is given, such as a human, car, or screenshot, the neural network will still produce one of the 10 date classes.

To reduce this problem, the web app includes:

- confidence thresholding
- simple date-like image precheck

However, this is not a perfect unknown-object detector.

A stronger future solution would be:

1. Train a separate date vs non-date detector.
2. Then classify date images into the 10 varieties.

Or add an 11th class:

```text
Unknown / Non-date
```

## How to Use the Trained Model

The main trained ResNet-50 checkpoint is stored in:

```text
checkpoints/best_resnet50_finetuned.pt
```

The custom CNN checkpoint is stored in:

```text
checkpoints/best_custom_cnn.pt
```

The model automatically uses CUDA if available, otherwise CPU:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

This allows training on a GPU machine and running inference on a CPU laptop.

For detailed local environment and conda commands, see:

```text
conda_config_cmds.txt
```

## Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet-50
- Custom CNN
- Flask
- NumPy
- Pillow
- Matplotlib
- Jupyter Notebook
- Miniconda / Anaconda

## Course Information

**Course Instructor / Supervisor**  
[Prof. Syed Muhammad Naqi](https://cs.qau.edu.pk/profiles/naqi.htm)

## Authors

- Saqib Bedar
- Hamza Khan Tariq

## License

This project is intended for academic and educational purposes.
