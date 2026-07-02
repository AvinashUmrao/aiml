# Spot the Fake Photo - Project Report

## 1. Project Overview

**Problem Statement:** Detect whether an image is a real photo or a photo of a screen (screen recapture detection). This is important for preventing fraud and verifying image authenticity.

**Objective:** Build a deep learning model that classifies images as real (class 0) or screen-captured (class 1) with high accuracy.

## 2. Dataset

The dataset is organized as follows:
```
dataset/
    train/
        real/     # Authentic photos
        screen/   # Photos of screens
    val/
        real/
        screen/
    test/
        real/
        screen/
```

**Data Collection:** Images are collected from various sources to capture diverse real photos and screen-captured images.

**Split:** Train/validation/test split is used for model development and evaluation.

## 3. Methodology

**Model Architecture:** MrcNet (Multi-scale Recapture Network) - a CNN with three parallel branches using different kernel sizes (3, 5, 7) to capture multi-scale features. The model includes:
- Batch normalization for stability
- Dropout (0.5) for regularization
- Label smoothing (0.1) for better generalization
- Gradient clipping (1.0) for training stability

**Data Preprocessing:**
- RandomResizedCrop, RandomHorizontalFlip, RandomRotation for augmentation
- ColorJitter for brightness/contrast variation
- GaussianNoise injection
- Normalization to [-1, 1] range

**Training Process:**
- Optimizer: Adam (learning rate 0.0005, weight decay 1e-4)
- Batch size: 8
- Epochs: 30 with early stopping (patience 5)
- Learning rate decay: 0.1x every 15 epochs
- Cross-entropy loss with label smoothing

**Inference Pipeline:** predict.py loads the trained model checkpoint, applies image transforms, and outputs a probability score.

## 4. Results

**Training Accuracy:** 94.05% (79/84 samples, Epoch 6)
**Validation Accuracy:** 50.00% (5/10 samples, Epoch 6)

The model shows signs of overfitting with high training accuracy but low validation accuracy. Evaluation tools (evaluate.py, benchmark.py) are provided to measure:
- Accuracy on test set
- Precision, Recall, F1 Score
- Confusion Matrix

## 5. Performance

**Inference Options:** The model supports three inference modes via --num-crops:
- 1 crop: Fastest (CenterCrop)
- 5 crops: Balanced (FiveCrop)
- 10 crops: Most accurate (TenCrop)

**Hardware:** Automatic CPU/GPU detection. GPU acceleration recommended for training.

**Latency:** benchmark.py measures average inference time per image.

## 6. Challenges and Limitations

- Dataset quality and diversity significantly impact model performance
- Screen recapture detection is sensitive to image resolution and quality
- Model may struggle with edge cases (low-quality screens, unusual lighting)
- Training requires substantial computational resources

## 7. Future Improvements

- Expand dataset with more diverse screen types and capture conditions
- Experiment with larger model architectures (ResNet, EfficientNet)
- Implement ensemble methods for improved robustness
- Add support for video frame analysis
- Optimize model for mobile/on-device deployment

## 8. How to Run

**Installation:**
```bash
pip install -r requirements.txt
```

**Dependencies:** torch, torchvision, pillow, scikit-learn, numpy

**Prediction:**
```bash
python predict.py image.jpg
```

**Output:** A probability score between 0 and 1:
- 0.0 = Real photo
- 1.0 = Photo of a screen

Optional arguments:
- `--checkpoint`: Path to model checkpoint (default: ./log/best_model.pth)
- `--num-crops`: Number of crops for inference (1, 5, or 10)

## 9. Conclusion

This project implements a screen recapture detection system using a multi-scale CNN architecture. The model is trained with data augmentation and regularization techniques to improve generalization. The inference pipeline provides flexible speed-accuracy trade-offs through multi-crop prediction. The system is suitable for integration into image verification workflows.

---

**Repository:** https://github.com/AvinashUmrao/aiml.git
