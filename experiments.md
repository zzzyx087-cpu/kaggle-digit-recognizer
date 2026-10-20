# Experiment 1: Logistic Regression

Model: Linear(784, 10)

Optimizer: SGD
Learning Rate: 0.01
Epochs: 10

Final Training Loss: 0.3763
Validation Accuracy: 89.90%

Observations:
- Loss decreased consistently.
- The model learned meaningful patterns.
- The linear architecture may limit performance.

# Experiment 2: Multi-Layer Perceptron (MLP)

Model: Linear(784, 128) → ReLU → Linear(128, 10)
Hidden Layers: 1
Hidden Neurons: 128
Activation Function: ReLU
Optimizer: SGD
Learning Rate: 0.01
Epochs: 100
Final Training Loss: 0.0710
Validation Accuracy: 96.60%

Observations:
- Loss decreased consistently throughout training.
- The MLP significantly outperformed Logistic Regression.
- ReLU enabled the model to learn nonlinear patterns.
- Increasing training epochs from 10 to 100 improved accuracy from 91.60% to 96.60%.
- Increasing hidden neurons or adding more hidden layers may further improve performance.

# Experiment 3: MLP with 256 Hidden Neurons

Model: Linear(784, 256) → ReLU → Linear(256, 10)

Hidden Layers: 1
Hidden Neurons: 256
Activation Function: ReLU

Total Parameters: 203,530

Optimizer: SGD
Learning Rate: 0.01
Epochs: 100

Final Training Loss: Not Recorded
Validation Accuracy: 96.79%

Observations:
- Increasing hidden neurons from 128 to 256 improved validation accuracy.
- Accuracy increased from 96.60% to 96.79%.
- The improvement was relatively small despite nearly doubling the parameter count.
- More hidden neurons do not necessarily lead to substantially better generalization.
- Further experiments are needed to determine whether the improvement is consistent across random seeds.
# Experiment 4: Deep MLP with Two Hidden Layers

Model: Linear(784, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 10)

Hidden Layers: 2
Hidden Neurons: 256, 128
Activation Function: ReLU
Total Parameters: 235,146
Optimizer: SGD
Learning Rate: 0.01
Epochs: 100
Final Training Loss: Not Recorded
Validation Accuracy: 97.43%

Observations:
- Adding a second hidden layer improved validation accuracy from 96.79% to 97.43%.
- The deeper MLP achieved the highest validation accuracy among the four experiments.
- Increasing network depth may help the model learn more complex nonlinear representations.
- Further experiments with different random seeds are needed to confirm whether the improvement is consistent.

# Experiment 5: Convolutional Neural Network (CNN)

Model: Conv2d(1, 32, 3) → ReLU → MaxPool2d(2) → Conv2d(32, 64, 3) → ReLU → MaxPool2d(2) → Flatten → Linear(1600, 10)

Convolutional Layers: 2

Activation Function: ReLU

Optimizer: SGD

Learning Rate: 0.01

Epochs: 10

Final Training Loss: Not Recorded

Validation Accuracy: 96.98%

Observations:
- The CNN achieved 96.98% validation accuracy after 10 epochs.
- Convolutional layers extract spatial features from handwritten digit images.
- The CNN used fewer training epochs than the previous MLP experiments.
- The deep MLP achieved higher validation accuracy (97.43%), although it was trained for 100 epochs.


# Experiment 6: CNN with Adam Optimizer

Model: Conv2d(1, 32, 3) → ReLU → MaxPool2d(2) → Conv2d(32, 64, 3) → ReLU → MaxPool2d(2) → Flatten → Linear(1600, 10)

Convolutional Layers: 2

Activation Function: ReLU

Optimizer: Adam

Learning Rate: 0.001

Epochs: 10

Final Training Loss: 0.0139

Validation Accuracy: 98.52%

Observations:
- Replacing SGD with Adam improved validation accuracy from 96.98% to 98.52%.
- Adam uses adaptive learning rates for individual parameters.
- The model achieved higher accuracy without changing the CNN architecture.
- Extending training to 20 epochs in a separate experiment resulted in 98.50% validation accuracy, suggesting that additional epochs alone did not improve performance.


# Experiment 6 (Additional): CNN with Cosine Annealing Scheduler

Model: Conv2d(1, 32, 3) → ReLU → MaxPool2d(2) → Conv2d(32, 64, 3) → ReLU → MaxPool2d(2) → Flatten → Linear(1600, 10)

Convolutional Layers: 2

Activation Function: ReLU

Optimizer: Adam

Initial Learning Rate: 0.001

Learning Rate Scheduler: CosineAnnealingLR

T_max: 20

Minimum Learning Rate: 0.00001

Epochs: 20

Final Training Loss: 0.0014

Validation Accuracy: 98.64%

Observations:
- CosineAnnealingLR gradually reduced the learning rate during training.
- Validation accuracy increased from 98.52% to 98.64%.
- The final training loss decreased to 0.0014.
- Learning rate scheduling may help refine parameter updates during later training stages.
- Lower training loss does not necessarily guarantee higher validation accuracy.


# Experiment 7: CNN with Batch Normalization

Model: Conv2d(1, 32, 3) → BatchNorm2d(32) → ReLU → MaxPool2d(2) → Conv2d(32, 64, 3) → BatchNorm2d(64) → ReLU → MaxPool2d(2) → Flatten → Linear(1600, 10)

Convolutional Layers: 2

Batch Normalization Layers: 2

Activation Function: ReLU

Optimizer: Adam

Learning Rate: 0.001

Epochs: 20

Final Training Loss: 0.0020

Validation Accuracy: 98.69%

Observations:
- Added BatchNorm2d after both convolutional layers.
- Batch Normalization standardizes intermediate feature activations and can improve training stability.
- Validation accuracy increased from 98.52% to 98.69%.
- The improvement was relatively small compared with the Adam baseline.
- Further experiments are needed to determine whether the improvement is consistent across random seeds.


# Experiment 8: CNN with Data Augmentation

Model: Conv2d(1, 32, 3) → ReLU → MaxPool2d(2) → Conv2d(32, 64, 3) → ReLU → MaxPool2d(2) → Flatten → Linear(1600, 10)

Convolutional Layers: 2

Activation Function: ReLU

Optimizer: Adam

Learning Rate: 0.001

Epochs: 20

Data Augmentation: RandomAffine

Rotation: ±10 degrees

Translation: 8% of image dimensions

Final Training Loss: 0.0418

Validation Accuracy: 99.10%

Observations:
- Applied random rotations and translations to training images to increase data diversity.
- Validation accuracy improved from 98.52% to 99.10%, achieving the first result above 99%.
- The final training loss was higher than in previous experiments, but validation accuracy improved significantly.
- Data augmentation encourages the model to learn more robust features instead of relying on fixed image positions and orientations.
- Lower training loss does not necessarily indicate better generalization performance.
- Data augmentation produced the largest validation accuracy improvement among the tested optimization methods.
- Further experiments with different random seeds are needed to confirm whether the improvement is consistent.

# Experiment 9: CNN with Combined Optimization

Model: CNN with Batch Normalization
Optimizer: Adam
Learning Rate: 0.001
Scheduler: CosineAnnealingLR
Data Augmentation: RandomAffine
Epochs: 20
Final Training Loss: 0.0290
Best Validation Accuracy: 99.12%
Kaggle Public Accuracy: 99.246%

Observations:
- Combined Batch Normalization, data augmentation, and learning rate scheduling.
- Achieved the highest observed validation accuracy of 99.12%.
- Improved slightly over the previous augmentation experiment (99.10%).
- Achieved a Kaggle Public Score of 0.99246.
- The small validation improvement does not establish statistically significant superiority.