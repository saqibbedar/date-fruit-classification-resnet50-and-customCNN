# Date Fruit Classification using ResNet-50

A computer vision project for classifying date fruit varieties using **ResNet-50 transfer learning** with **PyTorch**.
The project includes model training, fine-tuning, evaluation, confusion matrix analysis, and a simple **Flask web application** for image upload and prediction.

## Project Overview

This project classifies date fruit images into 10 varieties:

* Ajwa
* Aseel
* Fasli-Toto
* Galaxy
* Medjool
* Meneifi
* Nabtat-Ali
* Shaishe
* Sokari
* Sugaey

The model was trained using transfer learning with a pretrained ResNet-50 backbone.
The final classifier layer was replaced to predict 10 date fruit classes.

## Features

* ResNet-50 transfer learning
* Frozen-backbone baseline training
* Fine-tuning of ResNet `layer4`
* Train/validation/test split
* Image preprocessing and augmentation
* Confusion matrix evaluation
* Precision, recall, and F1-score analysis
* Flask web app for image upload and prediction
* Confidence thresholding
* Basic date-like image precheck for rejecting obvious non-date inputs

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

* 70% training
* 15% validation
* 15% testing

## Model

Architecture:

* Backbone: ResNet-50
* Pretrained weights: ImageNet
* Modified classifier head:

```python
Dropout(p=0.30)
Linear(2048, 10)
```

Training strategy:

1. Freeze ResNet-50 backbone and train only the classifier head.
2. Load the best frozen model.
3. Unfreeze `layer4` and classifier head.
4. Fine-tune with smaller learning rates.

## Results

### Frozen-Backbone Training

Best validation accuracy:

```text
88.86%
```

### Fine-Tuned Model

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

## Confusion Matrix Observation

The model performs very well on most classes such as Ajwa, Medjool, Shaishe, Aseel, and Fasli-Toto.

The most challenging class is **Meneifi**, which is sometimes confused with:

* Medjool
* Sugaey
* Galaxy

This shows that some date varieties are visually similar and require deeper error analysis or additional data improvement.

## Project Structure

```text
date-fruit-classification-resnet50/
│
├── assets/
│   └── resnet_architecture.webp
│
├── checkpoints/
│   ├── best_resnet50_frozen.pt
│   └── best_resnet50_finetuned.pt
│
├── datasets/
│   ├── full/
│   ├── train/
│   ├── val/
│   └── test/
│
├── app.py
├── index.html
├── main.ipynb
├── main.py
├── pyproject.toml
├── requirements-demo-cpu.txt
├── requirements-dev.txt
└── README.md
```

## Installation

### Option 1: CPU Demo Setup

Use this option on a laptop without GPU.

```bash
python -m venv .venv
```

Activate environment on Windows:

```bash
.\.venv\Scripts\activate
```

Install CPU dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
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

### Option 2: GPU Training Setup

Use this option on a system with NVIDIA GPU and CUDA-supported PyTorch.

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install flask numpy pillow matplotlib ipykernel
```

## Running the Web App

Start the app:

```bash
python app.py
```

Then upload an image using the browser interface.

The app shows:

* predicted class
* confidence score
* top-k predictions
* accepted/rejected decision
* simple input quality check

## Important Note about Unknown Images

This is a closed-set classifier.

That means the model was trained only on 10 known date classes.
If a non-date image is given, such as a human, car, or screenshot, the neural network will still produce one of the 10 date classes.

To reduce this problem, the web app includes:

* confidence thresholding
* simple date-like image precheck

However, this is not a perfect unknown-object detector.

A stronger future solution would be:

1. Train a separate date vs non-date detector.
2. Then classify date images into the 10 varieties.

Or add an 11th class:

```text
Unknown / Non-date
```

## How to Use the Trained Model

The trained model checkpoint is stored in:

```text
checkpoints/best_resnet50_finetuned.pt
```

The model automatically uses CUDA if available, otherwise CPU:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

This allows training on a GPU machine and running inference on a CPU laptop.

## Technologies Used

* Python
* PyTorch
* Torchvision
* ResNet-50
* Flask
* NumPy
* Pillow
* Matplotlib
* Jupyter Notebook

## Authors

* Saqib Bedar
* Hamza Khan Tariq

## License

This project is intended for academic and educational purposes.
