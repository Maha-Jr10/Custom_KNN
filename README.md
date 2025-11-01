# 🤖 TP1: K-Nearest Neighbors (KNN) Algorithm

## 🎯 Project Overview

This repository contains the practical work (**TP1**) dedicated to understanding, implementing, and evaluating the **K-Nearest Neighbors (KNN)** classification and regression algorithm.

The main objective is to grasp the functioning of the KNN algorithm, apply it to a real-world dataset (like **Iris** or **Breast Cancer**), and compare its performance under different conditions and against other machine learning models.

The exercises are implemented in the provided Jupyter Notebook, `TP1_John_Muhammed.ipynb`.

---

## 📂 Repository Structure

* `app`: the folder containing an interactive streamlit application for learning purpose of  the KNN algorithm.
* `iris.csv`: the iris dataset use for the studies.
* `TP1_John_Muhammed.ipynb`: The primary Jupyter Notebook containing the full implementation, code, results, and analysis for all exercises of the TP.
* `TP 1 Partie 2 KNN.pdf`: The official problem statement/lab sheet detailing the exercises.

---

## ✨ Key Topics Covered

The project addresses various aspects of the KNN algorithm, including:

1.  **Data Preparation**: Splitting data (`train_test_split`) and **Standard Scaling** to manage KNN's sensitivity to data magnitude.
2.  **Model Optimization**: Determining the optimal value of **k** and visualizing the "k vs. precision" curve.
3.  **Weighted KNN**: Modifying the algorithm to **weight neighbor votes** based on distance (e.g., $1/\text{distance}^2$).
4.  **Decision Boundaries**: Visualizing the **decision frontiers** of the KNN on a 2D dataset.
5.  **Model Comparison**: Comparing KNN performance (precision/accuracy) with other models: **Decision Tree, SVM, and Logistic Regression**.
6.  **Error Analysis**: Identifying and analyzing **misclassified examples** and their nearest neighbors.
7.  **KNN for Regression**: Implementing a version of KNN where the prediction is the average (or weighted average) of the $k$ nearest neighbors.
8.  **Robust Evaluation**: Using **manual or cross-validation** (`cross_val_score`) for a more stable performance assessment.
9.  **Performance Optimization**: Utilizing **KDTree/BallTree** structures from `sklearn.neighbors` to accelerate neighbor search.
10. **Impact of Noise**: Studying the effect of adding **feature noise or label noise** on KNN's accuracy.

---

## 🛠 Prerequisites

To run the Jupyter Notebook, you will need to have Python and the following libraries installed:

* **Python** (3.7+)
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **scikit-learn** (`sklearn`)

### Installation

You can install the required libraries using `pip`:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
````

-----

## 🚀 How to Run

1.  **Clone the repository** (if this were a git repo):
    ```bash
    git clone [repository_url]
    cd tp1-knn-project
    ```
2.  **Launch the Jupyter Notebook:**
    ```bash
    jupyter notebook TP1_John_Muhammed.ipynb
    ```
3.  **Execute the cells** sequentially to run the code, generate the results, and see the visualizations.

-----

## How to run the app
visit the app folder and there is a dedication notebook with instructions on how to run the application.

-----

## 👤 Author

  * **Student:** John Muhammed
  * **Supervisor:** Pr. N. EL AKKAD
  * **Course:** Machine Learning

