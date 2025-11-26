
# ML-Implementations-From-Scratch

![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-1.24%2B-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Data_Viz-orange?logo=matplotlib&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

## Project Description

**ML-Implementations-From-Scratch** is a collection of Python implementations that peel back the layers of abstraction found in libraries like TensorFlow or Scikit-Learn. By building algorithms like K-Means, Neural Networks, and Transformers using primarily **NumPy**, this project serves as a practical guide to understanding the underlying mathematics—gradients, matrix multiplications, and probability distributions—that drive modern AI.

## Detailed Modules & Use Cases

### 📂 Work 1: K-Nearest Neighbors (KNN)

  * **Description:** Implements the KNN algorithm, a non-parametric method used for classification and regression. It predicts the label of a data point by looking at the 'K' closest labeled data points in the feature space.
  * **Key Files:** `knn.py`, `data.py`.
  * **Use Case:**
      * **Medical Diagnosis:** The included `heart_disease.csv` example demonstrates predicting the presence of heart disease based on patient metrics (age, cholesterol, etc.) by comparing them to similar historical patient profiles.

### 📂 Work 2: Linear Regression

  * **Description:** A fundamental algorithm for modeling the relationship between a scalar response and one or more explanatory variables. It uses Gradient Descent to minimize the error (cost function) and find the best-fitting line.
  * **Key Files:** `linear_regression.py`, `linear_regression_test.py`.
  * **Use Case:**
      * **Quality Prediction:** The project uses `winequality-white.csv` to predict the quality score of a wine based on chemical properties like acidity, sugar, and pH levels.

### 📂 Work 3: Binary & Multiclass Classification

  * **Description:** Explores linear classifiers for separating data into categories. It covers binary classification (two classes) and extends to multiclass problems using techniques like One-vs-All or Softmax regression.
  * **Key Files:** `classification.py`, `bm_classify.py` (likely Block Coordinate Descent or similar optimization).
  * **Use Case:**
      * **Handwritten Digit Recognition:** Uses a subset of the MNIST dataset (`mnist_subset.json`) to classify images of handwritten digits (0-9) into their respective numeric categories.

### 📂 Work 4: Neural Networks (MLP)

  * **Description:** A complete implementation of a Multi-Layer Perceptron (MLP). It features forward propagation (computing predictions) and backpropagation (computing gradients) to train the network weights from scratch.
  * **Key Files:** `neural_networks.py`, `runme.py`.
  * **Use Case:**
      * **Complex Pattern Recognition:** Capable of solving non-linear problems that simple linear classifiers cannot, such as recognizing complex shapes or patterns in the digits dataset.

### 📂 Work 5: Decision Trees & Boosting

  * **Description:** Implements Decision Trees which split data based on feature values to maximize information gain. It also includes **AdaBoost** (`boosting.py`), an ensemble technique that combines multiple "weak" decision trees to create a robust "strong" classifier.
  * **Key Files:** `decision_tree.py`, `boosting.py`.
  * **Use Case:**
      * **Robust Classification:** Ideal for tabular data where interpretability is key. AdaBoost is particularly useful for improving prediction accuracy on difficult datasets by focusing on previously misclassified instances.

### 📂 Work 6: K-Means Clustering

  * **Description:** An unsupervised learning algorithm that partitions data into 'K' distinct clusters based on distance to centroids. The algorithm iteratively refines the centroid positions.
  * **Key Files:** `kmeans.py`, `data_loader.py`.
  * **Use Case:**
      * **Image Compression:** The project demonstrates using K-Means on `baboon.tiff` to reduce the number of unique colors in an image. By clustering pixel colors and replacing them with the cluster center, the image size is significantly reduced.

### 📂 Work 7: Principal Component Analysis (PCA)

  * **Description:** A dimensionality reduction technique that projects data onto a lower-dimensional space while preserving the maximum variance. It is crucial for feature extraction and visualization.
  * **Key Files:** `pca.py`, `utils.py`.
  * **Use Case:**
      * **Word Embedding Analysis:** Used here to analyze word relationships (`analogy_task.txt`). By reducing high-dimensional word vectors to 2D or 3D, we can visualize semantic similarities (e.g., "King" is to "Queen" as "Man" is to "Woman").

### 📂 Work 8: Hidden Markov Models (HMM)

  * **Description:** A statistical Markov model with unobserved (hidden) states. It computes the probability of a sequence of observed events and is heavily used in temporal pattern recognition.
  * **Key Files:** `hmm.py`, `tagger.py`.
  * **Use Case:**
      * **Part-of-Speech (POS) Tagging:** The system reads sentences (`pos_sentences.txt`) and predicts the grammatical tag (Noun, Verb, Adjective) for each word based on the sequence context.

### 📂 Work 9: Transformers (Self-Attention)

  * **Description:** Implements the core architecture of modern NLP: the Transformer. It features the **Self-Attention** mechanism, allowing the model to weigh the importance of different words in a sentence regardless of their positional distance.
  * **Key Files:** `transformer_model.py`, `train.py`.
  * **Use Case:**
      * **Sequence Modeling:** This is the foundational architecture behind models like BERT and GPT, used here to learn dependencies in text sequences (`input.txt`) for tasks like text generation or translation.

### 📂 Work 10: Reinforcement Learning (RL)

  * **Description:** Focuses on agents taking actions in an environment to maximize cumulative reward. It implements **Q-Learning** and models problems as **Finite Markov Decision Processes (MDPs)**.
  * **Key Files:** `q_learning.py`, `finite_mdp.py`, `playground.ipynb`.
  * **Use Case:**
      * **Game Solving & Navigation:** The agent learns to navigate a "Grid World" environment, figuring out the optimal path to a goal state while avoiding penalties, purely through trial-and-error interaction.

## Installation & Running

### 1\. Clone the Repository

```bash
git clone https://github.com/nikelroid/machine-learning-implementations.git
cd machine-learning-implementations
```

### 2\. Dependencies

It is recommended to use a virtual environment.

```bash
pip install numpy matplotlib pandas jupyter
```

### 3\. Running a Module

Navigate to any specific directory and run the python script.

```bash
# Example: Linear Regression
cd "work 2"
python linear_regression_test.py
```

## Contributing

If you spot an error in the calculus or want to add a new algorithm (e.g., GANs, LSTM), please submit a Pull Request. Ensure your code relies primarily on NumPy.

## License

Distributed under the MIT License.

## Contact

Project Maintainer - [GitHub Profile](https://www.google.com/search?q=https://github.com/nikelroid)
