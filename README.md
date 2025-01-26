# EMNIST Character Classification Using Neural Networks

This repository demonstrates the application of deep learning techniques for classifying handwritten characters from the **EMNIST Balanced dataset**. The project involves building and optimizing two neural network architectures: a Multilayer Perceptron (MLP) and a Convolutional Neural Network (CNN). The key focus is on experimentation with various hyperparameters and techniques to improve model performance.

---

## Dataset Overview
The **EMNIST Balanced** dataset, derived from the NIST Special Database 19, contains:

- **112,800 training samples**
- **18,800 testing samples**
- **47 balanced classes** (combining uppercase and lowercase variations to reduce misclassification errors)
- Images are in a **28x28 grayscale** format.

---

## Project Objectives
1. **Build Neural Networks**: Develop and optimize MLPs and CNNs for EMNIST classification.
2. **Hyperparameter Tuning**: Experiment with a range of parameters and techniques, including:
   - Adaptive learning rate schedulers
   - Activation functions (ReLU, Leaky ReLU, ELU)
   - Optimizers (Adam, RMSprop, SGD)
   - Regularization techniques (L1, L2, Dropout)
   - Batch normalization
3. **Model Comparison**: Evaluate and compare MLP and CNN performance using metrics such as accuracy, precision, recall, and F1-score.
4. **Visualization**: Provide insights into training progress with loss/accuracy plots and confusion matrices.

---

## Implementation Steps

### 1. Data Preprocessing
- Imported the EMNIST dataset using PyTorch's dataset library.
- Visualized sample images and ensured proper data loading.
- Split the dataset into training and testing subsets using DataLoader.

### 2. Model Architectures
- **MLP**: Fully connected network with at least 3 hidden layers.
- **CNN**: Includes at least 2 convolutional layers.

### 3. Hyperparameter Exploration
- Tested different configurations:
  - **Learning Rate Schedulers**: Cosine Annealing, Step Decay
  - **Activation Functions**: ReLU, Leaky ReLU, ELU
  - **Optimizers**: Adam, RMSprop, SGD
  - **Regularization**: Explored L1 and L2 regularization, with/without Dropout
  - **Batch Normalization**: Compared models with and without batch normalization
- Performed cross-validation to determine the best combination of hyperparameters for each model.

### 4. Training and Evaluation
- Trained both MLP and CNN models using optimal configurations.
- Visualized the training progress through:
  - **Loss and Accuracy plots** over epochs
  - **Confusion Matrices** for predictions

### 5. Testing and Analysis
- Evaluated models on the test set and printed predictions for the top six samples.
- Compared MLP and CNN performance across:
  - Accuracy
  - Precision
  - Recall
  - F1-score

---

## Results
- Significant improvement in classification accuracy through hyperparameter optimization.
- Visualized comparative performance for MLP and CNN models using confusion matrices and evaluation metrics.

---

## Requirements
To run this project, ensure the following dependencies are installed:

```bash
pip install torch torchvision numpy pandas matplotlib
```

---

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/emnist-classification.git
   cd emnist-classification
   ```

2. **Run the script**:
   ```bash
   python main.py
   ```

3. **Output**:
   - Loss and Accuracy plots
   - Confusion matrices for MLP and CNN models
   - Top 6 predictions compared to true labels

---

## Repository Structure

```plaintext
emnist-classification/
|
|-- datasets/                 # Dataset processing scripts
|-- models/                   # MLP and CNN model architectures
|-- experiments/              # Hyperparameter tuning scripts
|-- outputs/                  # Saved models, logs, and visualizations
|-- utils.py                  # Helper functions
|-- main.py                   # Main script to run the project
|-- README.md                 # Project documentation
```

---

## Results Visualization

- **Loss and Accuracy Plots**:

![Loss and Accuracy Plot](outputs/loss_accuracy_plot.png)

- **Confusion Matrix (CNN)**:

![Confusion Matrix](outputs/confusion_matrix.png)

---

## Future Work
- Extend the project to other EMNIST splits (e.g., ByClass, ByMerge).
- Experiment with advanced architectures like ResNet or Transformer-based models for comparison.
- Deploy the best-performing model using a web application or API.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- EMNIST Dataset: [Kaggle - EMNIST](https://www.kaggle.com/datasets/crawford/emnist)
