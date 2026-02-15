# PyTorch MNIST Digit Classification 

A deep learning project that classifies handwritten digits (0-9) from the **MNIST dataset** using a Convolutional Neural Network (CNN) based on the **TinyVGG architecture**.

The model achieves **over 99% accuracy** on the test dataset.

## Project Overview
- **Goal:** Accurately classify grayscale handwritten digit images into 10 classes (0-9).
- **Architecture:** TinyVGG (CNN with Conv2d, ReLU, and MaxPool2d layers).
- **Framework:** PyTorch.
- **Performance:** ~99.0% Accuracy on the Test Set.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/BerkCpro/PyTorch-MNIST-Classifier.git](https://github.com/BerkCpro/PyTorch-MNIST-Classifier.git)
   cd PyTorch-MNIST-Classifier
   
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
To train the model and evaluate it on the test set, simply run the main script:
   ```bash
   python mnist_tinyvgg_classifier.py
```
The script will:

   1. Download the MNIST dataset
   2. Train the TinyVGG model for 10 epochs.
   3. Display training and testing loss/accuracy metrics.
   4. Save the trained model to the models/ directory.
   5. Generate and save a Confusion Matrix to the results/ directory.

# Results
The model demonstrates high precision and recall across all digit classes.

# Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

