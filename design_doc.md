#Kaggle Digit Recognizer
## 1.project goal
Build a handwritten digit recognition model using PyTorch.
The model should classify grayscale images into digits 0–9.

## 2. Dataset
Source: Kaggle Digit recognizer

Training data:
- 42,000 labeled images
- 784 pixels per image
- Image size: 28 × 28

Test data:
- 28,000 unlabeled images

## 3. Data Processing
1. Load CSV files using Pandas.

2. Separate features (X) and labels (y).

3. Normalize pixel values from 0–255 to 0–1.

4. Prepare input shapes:
   - MLP: (N, 784)
   - CNN: (N, 1, 28, 28)
5. Split training data into training and validation sets.

## 4. Models
Baseline:
- Logistic Regression
Model 1:
- MLP (Multi-Layer Perceptron)
Model 2:
- CNN (Convolutional Neural Network)

## 5. Training
Framework: PyTorch
Loss Function:
- CrossEntropyLoss
Optimizer:
- Adam
Initial Learning Rate:
- 0.001

## 6. Evaluation
Primary Metric:
- Validation Accuracy
Additional Analysis:
- Training Loss
- Validation Loss
- Misclassified Images

## 7. Final Output
- Trained CNN model
- submission.csv
- Experiment results
- GitHub README

## 8. Success Criteria
1. Independently implement the training pipeline.
2. Train and evaluate MLP and CNN models.
3. Aim for approximately 98% CNN validation accuracy.
4. Successfully submit predictions to Kaggle.
5. Explain the model architecture and training process.