Dual-ANN Framework: Classification & Regression with Optuna
1. Project Overview
This repository contains two distinct Deep Learning projects that demonstrate the power of Artificial Neural Networks (ANN) in solving different business problems.
Both projects utilize Optuna for automated hyperparameter tuning to ensure peak model performance.

Project 1: Employee Churn Predictor (Classification) – A model designed to predict whether an employee will stay or leave the company based on HR metrics.

Project 2: Salary Estimation Predictor (Regression) – A predictive model that estimates an employee's salary by analyzing demographic data and professional experience.

2. Technical Stack
Framework: TensorFlow, Keras

Optimization: Optuna (Hyperparameter Tuning)

Preprocessing: Scikit-learn (StandardScaler, LabelEncoder, train_test_split)

Data Analysis: Pandas, NumPy

Performance Metrics: Gini Coefficient, AUC, R2 Score, MAE

3. Project 1: Employee Churn Analysis (Classification)
This project focuses on identifying risk factors in employee retention.

Data Engineering:
Categorical Handling: Used pd.get_dummies with drop_first=True to eliminate multicollinearity.

Feature Scaling: Implemented StandardScaler to ensure the neural network processes all inputs on the same scale.

Architecture & Optimization:
ANN Design: A multi-layer network with a Sigmoid output layer for binary classification.

Optuna Objective: Maximize the AUC-ROC score.

Automated Search: Tuned the number of neurons (6–32), optimizer types (Adam, SGD, RMSprop, Adagrad), and the learning rate on a logarithmic scale.

Evaluation: The final model was evaluated using the Gini Coefficient (Gini = 2 * AUC - 1), reaching a strong performance of ~78% Gini on the test set.

4. Project 2: Salary Estimation (Regression)
A regression-based ANN focused on predicting continuous numerical values for salaries.

Advanced Feature Engineering:
Experience Metrics: Created custom features like Job_Avg_Exp and Edu_Avg_Exp to capture the average years of experience based on job titles and education levels.

Data Cleaning: Handled inconsistent labeling in "Education Level" (e.g., standardizing
"Master's" and "Master's Degree") and encoded categorical features using LabelEncoder.

Architecture & Optimization:
ANN Design: A regression-oriented network with a ReLU activation in the output layer to ensure non-negative salary predictions.

Optuna Objective: Maximize the R2 Score.

Success Metric: Achieved an R2 Score of ~72%, indicating that the model successfully explains a large portion of the variance in salary data.

5. Automated Tuning Pipeline
Both projects share a standardized Optuna logic that automates the "trial and error" process:

def create_model(trial):
    model = Sequential()
    # Dynamic layer sizing
    model.add(Dense(units=trial.suggest_int('units_layer1', 6, 32), activation='relu'))
    # Dynamic optimizer selection
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop', 'adagrad'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    return model

  7. Conclusion
  These projects highlight the flexibility of Artificial Neural Networks when combined with intelligent optimization tools like Optuna.
  By automating hyperparameter search and implementing rigorous feature engineering, 
  we achieved high-accuracy models for both categorical and numerical prediction tasks.
