# KNN Classifier — Iris Flower Species Prediction

A supervised machine learning project that implements and tunes a **K-Nearest Neighbors (KNN)** classifier to predict the species of iris flowers based on physical measurements.

---

## Overview

This notebook walks through the full ML workflow:

1. **Libraries** — import all required dependencies
2. **Data Extraction** — load and preview the Iris dataset
3. **EDA** — explore class balance, feature distributions, and pairwise relationships
4. **Model Building** — build a pipeline (StandardScaler → KNN) and tune hyperparameters via GridSearchCV
5. **Evaluation** — assess performance using accuracy, precision, recall, F1-score, and a confusion matrix

---

## Dataset

The dataset (`iris_dataset.csv`) contains **150 samples** from 3 iris species:

| Feature | Description |
|---------|-------------|
| `sepal_length` | Sepal length in cm |
| `sepal_width` | Sepal width in cm |
| `petal_length` | Petal length in cm |
| `petal_width` | Petal width in cm |
| `species` | Target class: *setosa*, *versicolor*, *virginica* |

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Model, pipeline, tuning, metrics |
| `scipy` | Statistical utilities |

---

## How to Run

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the notebook:
   ```bash
   jupyter notebook
   ```

5. Open `knn_iris.ipynb` and run all cells.

---

## Model Details

- **Algorithm:** K-Nearest Neighbors (`KNeighborsClassifier`)
- **Preprocessing:** `StandardScaler` (feature normalization — essential for distance-based algorithms)
- **Hyperparameter tuning:** `GridSearchCV` over `n_neighbors` (odd values 1–10) and `weights` (`uniform` / `distance`)
- **Cross-validation:** 10-fold `StratifiedKFold`
- **Scoring metric:** Weighted F1-Score

---

## Results

After grid search and cross-validation, the best model is evaluated on a held-out test set (30% of the data). Metrics reported: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

---

## Project Structure

```
├── knn_iris.ipynb       # Main notebook
├── iris_dataset.csv     # Dataset
├── requirements.txt     # Python dependencies (optional)
└── README.md            # Project documentation
```

---

## License

This project is intended for educational purposes.
