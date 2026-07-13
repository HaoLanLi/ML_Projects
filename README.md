# Exercise#1 California Housing Price Prediction 🏠

## Introduction

This is my first Machine Learning exercise while learning from **"Zero-Basic Machine Learning" by Jia Huang**.

I completed this project on **Day 5 of my Machine Learning journey**.

The purpose of this project is to learn the basic workflow of a Machine Learning project by using **Linear Regression** to predict California housing prices.

Through this exercise, I practiced:

- Loading data
- Preparing features and labels
- Splitting training and testing data
- Training a regression model
- Making predictions
- Evaluating model performance
- Visualizing prediction results

---

## Project Goal

The goal of this project is to predict California house prices based on housing-related features.

This is a **supervised learning regression problem**.

The model learns from historical housing data and predicts:

```
median_house_value
```

---

## Dataset

The dataset used in this project is the California Housing Dataset.

The data contains information about different housing areas in California.

Example features include:

| Feature | Description |
|---------|-------------|
| median_income | Median income of residents |
| housing_median_age | Median age of houses |
| total_rooms | Total number of rooms |
| total_bedrooms | Total number of bedrooms |
| population | Population of the area |
| households | Number of households |
| latitude | Latitude location |
| longitude | Longitude location |

Target:

`median_house_value`

The median value of houses in each area.

---

## Machine Learning Workflow

The workflow of this project:

```
Load Dataset
      ↓
Select Features and Target
      ↓
Split Training and Testing Data
      ↓
Train Linear Regression Model
      ↓
Make Predictions
      ↓
Evaluate Model
      ↓
Visualize Results
```

---

## Model

### Linear Regression

For this project, I used **Linear Regression**.

Linear Regression tries to find the relationship between input features and the target value.

The model learns:

```
Housing Features → House Price
```

and uses this relationship to predict unknown house prices.

---

## Implementation

The main steps include:

### 1. Load Dataset

Used Pandas to read the housing dataset from a CSV file.

```python
pd.read_csv()
```

---

### 2. Prepare Data

Separated:

- Features (X)
- Target value (y)

```python
x = df_housing.drop("median_house_value", axis=1)

y = df_housing.median_house_value
```

---

### 3. Train/Test Split

Split the dataset into:

- 80% training data
- 20% testing data

using Scikit-learn:

```python
train_test_split()
```

---

### 4. Train Model

Created and trained the Linear Regression model:

```python
LinearRegression()

model.fit(x_train, y_train)
```

---

### 5. Prediction

Used the trained model to predict house prices:

```python
model.predict(x_test)
```

---

## Visualization

I created a scatter plot to compare:

- Actual house prices
- Predicted house prices

The visualization helps understand how well the Linear Regression model fits the data.

---

## Result

The model was evaluated using the R² score:

```python
model.score(x_test, y_test)
```

The score represents how well the model can explain the relationship between the input features and house prices.

---

## What I Learned

From this first ML exercise, I learned:

- How to use Pandas for data loading and manipulation
- The difference between features and target variables
- How training data and testing data work
- How Linear Regression performs predictions
- How to evaluate a basic ML model
- How visualization helps understand model performance

This project helped me understand the basic idea behind Machine Learning:

```
Data → Training → Model → Prediction → Evaluation
```

---

## Future Learning Goals

For my next Machine Learning projects, I want to learn:

- More regression algorithms
- Data preprocessing techniques
- Feature engineering
- Model comparison
- Deep Learning and Computer Vision

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## Project Status

✅ Completed

This is my first step into Machine Learning, and I will continue building more projects while improving my understanding of ML concepts.