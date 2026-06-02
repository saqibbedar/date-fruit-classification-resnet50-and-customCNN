# Date Fruit Classification Dataset + ResNet-50 Project

This Kaggle package contains a complete computer vision project for classifying date fruit varieties using ResNet-50 transfer learning.

It includes:

* A 10-class date fruit image dataset
* The exact train/validation/test split manifest used in the project
* Fine-tuned ResNet-50 PyTorch checkpoints
* Source code
* Jupyter notebook
* Flask web app for local image upload and prediction

## Project Authors / Maintainers

* Saqib Bedar
* Hamza Khan Tariq

## Classes

The dataset contains 10 date fruit varieties:

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

## Dataset Size

The dataset contains 7,000 images.

Each class contains 700 images.

## Experimental Split

The project used the following split:

* 70% training
* 15% validation
* 15% testing

Approximate split size:

* Training: 4,890 images
* Validation: 1,050 images
* Test: 1,060 images

The exact split is provided in:

```text
splits/split_manifest.csv
```

A class-wise split summary is provided in:

```text
splits/split_summary.csv
```

## Included Model Checkpoints

The package includes trained PyTorch model checkpoints:

```text
checkpoints/best_resnet50_frozen.pt
checkpoints/best_resnet50_finetuned.pt
```

The final model uses:

* Backbone: ResNet-50
* Pretrained weights: ImageNet
* Classifier head: Dropout + Linear layer
* Task: 10-class date fruit image classification

## Reported Results

Frozen-backbone validation accuracy:

```text
88.86%
```

Fine-tuned validation accuracy:

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

## Source Code

The `source/` folder contains:

* Training notebook
* Python scripts
* Flask web app
* Project README
* Environment metadata

The complete GitHub project is available here:

```text
https://github.com/saqibbedar/date-fruit-classification-resnet50
```

## Dataset Source and Rights Disclaimer

The image data was collected from publicly available internet sources and organized for an academic computer vision semester project.

The project authors do not claim ownership of the original images.

All image rights remain with their respective owners.

This package is shared for educational, academic, and research purposes only.

If you are the owner of any image included in this dataset and want it removed, please contact the maintainers through the linked GitHub repository.

Because the original image rights are not fully verified, users should perform their own due diligence before using this dataset for commercial, redistributed, or production purposes.

## Known Limitations

* The dataset mostly contains controlled date fruit images.
* Many images have clean or white backgrounds.
* The trained model is a closed-set classifier.
* The model may produce overconfident predictions on non-date images.
* Some classes visually overlap, especially Meneifi, Medjool, and Sugaey.
* The model should not be treated as a production-grade food grading system without further validation.

## Recommended Use

This package is suitable for:

* computer vision learning
* transfer learning practice
* PyTorch ResNet-50 fine-tuning
* multiclass image classification
* confusion matrix analysis
* image preprocessing experiments
* academic demonstrations

## Not Recommended For

This package is not recommended for:

* commercial use without image rights verification
* production food grading systems
* real-world deployment without further validation
* unknown-object detection without additional non-date training data
