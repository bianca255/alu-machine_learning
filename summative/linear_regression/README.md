# Linear Regression Summative

## Mission and Problem
Empower underprivileged children through technology and digital education so they can thrive in a technology-driven world.
Predict student final math outcomes (G3) early using learning and socio-demographic signals.
Enable schools and community programs to identify at-risk learners before final exams.
Support targeted, data-driven interventions that improve learning outcomes and long-term opportunity.

## Dataset
- Name: UCI Student Performance (Math)
- Source: https://www.kaggle.com/datasets/whenamancodes/student-performance
- File used: `student-mat.csv`
- Size: 395 rows, mixed categorical and numeric features
- Regression target: `G3` (final math grade)

## Project Structure
linear_regression_model/
|-- summative/
|   |-- linear_regression/
|   |   |-- multivariate.ipynb
|   |   |-- predict_best_model.py
|   |-- API/
|   |   |-- .gitkeep
|   |-- FlutterApp/
|       |-- .gitkeep
|-- requirements.txt
|-- README.md

## What The Notebook Covers
- Data loading and exploratory analysis
- Two meaningful visualizations (target distribution and correlation heatmap)
- Feature engineering with interpretation (column dropping and encoding decisions)
- Numeric conversion and standardization with scikit-learn preprocessing
- Linear regression with gradient descent (SGD), Decision Tree, and Random Forest
- Train/test loss curve plotting and before-vs-after fitted linear scatter visualization
- Best-model selection by lowest loss and model export for Task 2

## Outputs
- Saved best model: `best_model.pkl`
- Saved scaler: `scaler.pkl`
- Saved feature list: `feature_columns.json`


## Quick Run
1. Install dependencies: `pip install -r requirements.txt`
2. Download `student-mat.csv` from Kaggle and place it inside `summative/linear_regression/`
3. Open and run all cells in `summative/linear_regression/multivariate.ipynb`
4. Test saved model prediction with your payload:
	`python summative/linear_regression/predict_best_model.py --input payload.json`
