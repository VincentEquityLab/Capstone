# Capstone
Berkeley ML &amp; AI Capstone Project-Real Estate



📁 Dataset Overview

# House Price Prediction - Capstone Project

Capstone Project for **Berkeley HAAS (ML & AI Program)**

---

## ❓ What question are we trying to answer?
- Employ machine learning to **predict house prices** in King County, USA.

## 📃 What kind of problem is it?
- **Regression** problem

---

## 📁 Dataset Overview

- **Original dataset** contains ~21,000 rows and 21 columns (features + target)

### Features Description

| #  | Column Name     | Data Type | Description                                        |
|----|------------------|-----------|----------------------------------------------------|
| 1  | id               | integer   | Unique ID for each home sold                      |
| 2  | date             | string    | Date the house was sold                           |
| 3  | price            | float     | ✨ Target variable: house price                 |
| 4  | bedrooms         | integer   | Number of bedrooms                                |
| 5  | bathrooms        | float     | Number of bathrooms                               |
| 6  | sqft_living      | integer   | Living area in square feet                        |
| 7  | sqft_lot         | integer   | Lot size in square feet                           |
| 8  | floors           | float     | Number of floors in the house                     |
| 9  | waterfront       | integer   | Has waterfront view (0 or 1)                      |
| 10 | view             | integer   | Quality of the view                               |
| 11 | condition        | integer   | Condition of the house                            |
| 12 | grade            | integer   | Overall grade (workmanship & design)              |
| 13 | sqft_above       | integer   | Square feet above ground                          |
| 14 | sqft_basement    | integer   | Square feet in the basement                       |
| 15 | yr_built         | integer   | Year built                                        |
| 16 | yr_renovated     | integer   | Year renovated (0 if never renovated)             |
| 17 | zipcode          | integer   | Zip code area                                     |
| 18 | lat              | float     | Latitude                                          |
| 19 | long             | float     | Longitude                                         |
| 20 | sqft_living15    | integer   | Avg living area of nearest 15 neighbors           |
| 21 | sqft_lot15       | integer   | Avg lot size of nearest 15 neighbors              |

---

# 📊 Exploratory Data Analysis (EDA)

📂 **File**: `scripts/EDA.ipynb`

EDA focused on understanding the distribution and correlation of key variables, revealing patterns and potential insights for the regression task.

--- 
## 🧹 1. Cleaning
Verified null values: 
✅ No missing data found in any column.

Removed duplicate rows: 🔄 None detected.

Verified data types and converted date to datetime format.

Renamed some variables for consistency.



## 🔍2. Univariate & Bivariate Data Distributions

Below are visual insights generated during the Exploratory Data Analysis phase, helping us understand how each feature behaves individually and in relation to house prices.

**ID Distribution**  
![Distribution of ID](images/EDA-Distribution%20of%20ID.png)

**House Price Distribution**  
![Distribution of price](images/EDA-distribution%20of%20Price.png)

**Bedrooms Count Distribution**  
![Distribution of bedrooms](images/EDA-distribution%20of%20bedrooms.png)

**Bathrooms Count Distribution**  
![Distribution of bathrooms](images/EDA-distirbution%20of%20bathrooms.png)

**Living Area (sqft_living)**  
![Distribution of sqft_living](images/EDA-distirbution%20of%20sqt.png)

**Lot Size (sqft_lot)**  
![Distribution of sqft_lot](images/EDA-distirbution%20of%20sqft_lot.png)

**Floors (Levels per house)**  
![Distribution of floors](images/EDA-distribution%20of%20floors.png)

**Waterfront Indicator**  
![Distribution of waterfront](images/EDA-distirbution%20of%20waterfront.png)

**View Quality**  
![Distribution of view](images/EDA-distirbution%20of%20view.png)

**House Condition**  
![Distribution of condition](images/EDA-distribution%20of%20condition.png)

**House Grade**  
![Distribution of grade](images/EDA-distribution%20of%20grade.png)

**Above Ground Area (sqft_above)**  
![Distribution of sqft_above](images/EDA-distribution%20of%20sqt-above.png)

**Basement Area (sqft_basement)**  
![Distribution of sqft_basement](images/EDA-distribution%20of%20sqft-basement.png)

**Year Renovated**  
![Distribution of yr_renovated](images/EDA-distribution%20of%20yr_renovated.png)

**Living Area of Nearby Houses (sqft_living15)**  
![Distribution of sqft_living15](images/EDA-distribution%20of%20sqft-living15.png)

**Lot Size of Nearby Houses (sqft_lot15)**  
![Distribution of sqft_lot15](images/EDA-distirbution%20of%20sqdf-lot15.png)



---

## 📌3 Bivariate Visualizations (Relationships Between Features)
---


**📉 Price vs Bathrooms**  
More bathrooms tend to be associated with higher prices.  
![Price vs Bathrooms](images/EDA-price%20vs%20bathrooms.png)

**📉 Price vs Floors**  
More floors show a moderate impact on price.  
![Price vs Floors](images/EDA-price%20vs%20floors.png)

**📉 Price vs Grade**  
Better grades (higher build quality and finish) are strongly associated with higher prices.  
![Price vs Grade](images/EDA-price%20vs%20grade.png)

**📉 Price vs View**  
Houses with better views generally have higher prices.  
![Price vs View](images/EDA-price%20vs%20view.png)

**📉 Price vs Living Area (sqft_living)**  
Strong positive correlation—larger homes are more expensive.  
![Price vs sqft_living](images/EDA-privce%20vs%20sqft%20living.png)

**📉 Price vs Above Ground Area (sqft_above)**  
Above-ground area is a strong predictor of price.  
![Price vs sqft_above](images/EDA-prive%20vs%20sqt%20above.png)

---

### Geographic distribution of houses (colored by price)

- Houses with the highest prices are concentrated in specific clusters, mostly around water or upscale neighborhoods.
- The latitude-longitude map shows a clear price zoning pattern.

📍  
![House locations colored by price](images/EDA-House%20locations.png)

---

## 🧪 4. Multivariate Analysis

Multivariate Analysis involved visualizing one continuous variable (price) in relation to two or more categorical variables to identify deeper patterns.

---

### 1. Analysis of `sqft_living` across `bedrooms` and `price`

- Larger living areas correlate with higher prices, especially for homes with 3–5 bedrooms.
- Smaller homes (1–2 bedrooms) rarely exceed the \$1 million mark.

📊  
![Distribution of sqft_living](images/EDA-distirbution%20of%20sqt.png)

---

### 2. Analysis of `sqft_above` vs. `grade` and `price`

- Higher grade homes usually have larger above-ground area.
- Grade is a strong indicator of both size and price.

📊  
![Distribution of sqft_above](images/EDA-distribution%20of%20sqt-above.png)

---

### 3. Analysis of `yr_renovated` vs. `price` and `condition`

- Recently renovated homes have higher average prices.
- Most houses haven't been renovated (value = 0), yet those that are show a clear uplift in price.

📊  
![Distribution of yr_renovated](images/EDA-distribution%20of%20yr_renovated.png)

---

### 4. Analysis of `waterfront` and `view` vs. `price`

- Waterfront properties fetch significantly higher prices.
- Better views (scale 3–4) also increase house value sharply.

📊  
![Distribution of waterfront](images/EDA-distirbution%20of%20waterfront.png)  
![Distribution of view](images/EDA-distirbution%20of%20view.png)

---

### 5. Analysis of `grade` vs. `price` and `sqft_living`

- `grade` and `sqft_living` are the most positively correlated features with `price`.
- Most homes fall within grade 7–10.

📊  
![Distribution of grade](images/EDA-distribution%20of%20grade.png) 

### 6 Correlation Matrix
The heatmap below shows the Pearson correlation between numerical features.
Key insights:
![Correlation Matrix](images/EDA-Correlation%20Matrix.png) 
price is highly correlated with sqft_living and grade.

sqft_above also shows strong correlation with price.
#### 📈 Top 5 Correlated Features with Price

| Feature        | Correlation with Price |
|----------------|------------------------|
| sqft_living    | 0.70                   |
| grade          | 0.67                   |
| sqft_above     | 0.61                   |
| bathrooms      | 0.53                   |
| view           | 0.40                   |


## 5. 📦 Outliers

Outliers check was performed using Z-Score on key numerical fields such as `price`, `sqft_living`, and `sqft_lot`.  
No extreme outliers were removed, as most values fell within reasonable ranges.  
Some skewness was observed in `sqft_lot` and `price`, which were addressed via log transformation during **Feature Engineering**.

### 📊 Boxplots of Key Variables

**Boxplot – Bedrooms**  
![Boxplot - Bedrooms](images/boxplot%20bedrooms.png)

**Boxplot – Price**  
![Boxplot - Price](images/boxplot%20price.png)

---

### 📋 Summary Statistics of Key Features

| Feature           | Mean       | Std Dev   | Min     | 25%     | 50%     | 75%     | Max       |
|------------------|------------|-----------|---------|---------|---------|---------|------------|
| Price            | 540,088    | 367,127   | 75,000  | 321,950 | 450,000 | 645,000 | 7,700,000  |
| Sqft Living      | 2,079.9    | 918.4     | 290     | 1,427   | 1,910   | 2,550   | 13,540     |
| Sqft Lot         | 15,106.9   | 41,420.5  | 520     | 5,040   | 7,618   | 10,500  | 1,651,359  |
| Bedrooms         | 3.37       | 0.93      | 0       | 3       | 3       | 4       | 33         |
| Bathrooms        | 2.11       | 0.77      | 0       | 1.75    | 2.25    | 2.50    | 8.00       |
| Grade            | 7.66       | 1.18      | 1       | 7       | 7       | 8       | 13         |
| Yr Built         | 1971       | 29.37     | 1900    | 1951    | 1975    | 1997    | 2015       |


> 🔎 These values helped detect the need for normalization and inspired additional feature creation such as log-transformed price and flags for outliers in feature engineering.


---

## 🔚 Conclusion

- `sqft_living`, `grade`, and `bathrooms` are the most influential **numerical features** affecting house prices.
- **Categorical features** such as `waterfront`, `view`, and `renovation status` create distinct price segments.
- **Multivariate patterns** show that combinations like high `grade` + large `living space` + `waterfront` significantly boost property value.
- **Geographical clustering** of high-value homes suggests the importance of incorporating location-based features into modeling.

### 🔎 Insights Recap

- Homes with a **waterfront**, **higher grade**, or **better view** tend to be priced significantly higher.
- `sqft_living` and `grade` have the **strongest correlation** with price.
- **Outliers** were identified in features such as `sqft_living`, `sqft_lot`, and `price`, requiring transformation or capping during preprocessing.


---
## 🧠 Feature Selection 

📂 **File**: `scripts/Feature_Engineering.ipynb`

Identifying the most relevant features is critical to building effective predictive models. We used both **correlation analysis** and **model-based feature importance** to guide our selection.

---

### 🔗 Correlation with Price (Pearson)

We visualized how each numeric feature correlates with `price` using a heatmap.

![Correlation with Price](images/Correlation%20with%20Price.png)

> `sqft_living`, `grade`, and `sqft_above` have the strongest positive correlations with price.

---

### 🌳 Feature Importance from Decision Tree

Decision Tree Regressor helps highlight the most decisive splits used for price prediction.

![Decision Tree Feature Importance](images/features%20importances.png)

---

### 🌲 Feature Importance from Random Forest

Ensemble-based Random Forest model confirms the relevance of certain features.

![Random Forest Feature Importance](images/features%20importance%20from%20Random%20forest.png)

---

### ✅ Key Features Selected for Modeling

| Feature          | Justification                           |
|------------------|-----------------------------------------|
| `sqft_living`     | Strongest linear correlation with price |
| `grade`           | Most important in both tree models      |
| `sqft_above`      | Highly correlated and frequently used   |
| `bathrooms`       | Contributes to price variation          |
| `view`, `lat`     | Adds spatial & visual differentiation   |
| `waterfront`      | Clear segmenting factor for luxury      |
| `yr_built`        | Reflects construction quality/era       |

# Classification 
## 🤖 K-Nearest Neighbors (KNN) Regression

K-Nearest Neighbors (KNN) was used as a baseline model to predict housing prices. It is a simple, non-parametric method that relies on the similarity of neighboring data points.

---

### ⚙️ Methodology

- Features were scaled using **MinMaxScaler** before training.
- K values from **1 to 20** were tested using the **validation set**.
- Performance was evaluated using:
  - **R² Score** (explained variance)
  - **Root Mean Squared Error (RMSE)**

---

### 📈 R² Score vs. Number of Neighbors (K)

The R² score peaked around **K = 6**, indicating the best generalization capability at that point.

![KNN R² Score](images/KNN-R2%20Score.png)

---

### 📉 RMSE vs. Number of Neighbors (K)

RMSE reached its minimum (~**165,800**) when **K = 6**, suggesting it as the optimal number of neighbors.

![KNN RMSE vs K](images/KNN-RMSE%20vs%20K.png)

---

### 🔢 Key Metrics (Best Performing K = 6)

| Metric       | Value     |
|--------------|-----------|
| **K**        | 6         |
| **R² Score** | 0.787     |
| **RMSE**     | 165,800   |

---

### 🧠 Insights

- KNN achieved a decent **R² score of 0.787** at **K=6**.
- Performance plateaued after **K > 6**, with minimal variation in results.
- Although effective as a benchmark, KNN:
  - **Lacks interpretability**
  - **Scales poorly** with large datasets (due to distance computation)

## 🌳 Decision Tree Regressor

The **Decision Tree Regressor** was used to understand feature splits and gain interpretability into how the model predicts house prices.

### 🔢 Top 10 Feature Importances

The most influential features according to the Decision Tree model are:

- `grade` (overall construction & design quality)
- `sqft_living` (interior living space)
- `lat` (latitude – proxy for location)
- `long` (longitude)
- `waterfront` (whether the property has waterfront view)

📊 **Feature Importance Plot**  
![Top 10 Feature Importances - Decision Tree](images/top%2010%20feature%20importance%20-%20decision%20Tree.png)

---

### 🌲 Simplified Decision Tree (Max Depth = 2)

A simplified version of the trained tree reveals key decision rules learned by the model. The most frequent splits occurred on:

- `grade`
- `sqft_living`
- `lat`

🧠 **Decision Tree Visualization**  
![Decision Tree](images/Decision%20Tree.png)

## 🔍 Logistic Regression with PCA

To visualize the classification and assess performance, a logistic regression model was trained using PCA-reduced features (first 2 principal components). PCA helped reduce dimensionality and allowed us to plot the decision space.

### 📌 Confusion Matrix

This matrix evaluates the performance of the classifier:

- **True Positives (TP)**: 1721
- **True Negatives (TN)**: 1831
- **False Positives (FP)**: 339
- **False Negatives (FN)**: 429

Accuracy, precision, recall, and F1-score were computed based on this confusion matrix.

![Confusion Matrix](images/Confusion%20Matrix.png)

---

### 📊 PCA Projection of KC House Data

The PCA projection provides a 2D visualization of the data structure after dimensionality reduction. The model attempts to separate classes based on price category using logistic regression.

- Red: High-priced houses
- Blue: Low-priced houses

While overlap exists, we can see that PCA + Logistic Regression offers some level of linear separability.

![PCA Projection](images/PCA%20Projection.png)


## 🧬 Model Building & Evaluation

📂 **Scripts used**:
- `scripts/LinearRegression.ipynb`
- `scripts/TreeModels.ipynb`
- `scripts/XGBoost.ipynb`

### 🎯 Models Tested

- Linear Regression
- Ridge / Lasso Regression
- Decision Tree
- Random Forest
- XGBoost

### 📊 Metrics Used

- RMSE (Root Mean Squared Error)
- R² Score

### 📈 Model Comparison Table

| Model             | RMSE     | R² Score |
|------------------|----------|------------|
| Linear Regression | 216,000  | 0.70       |
| Decision Tree     | 163,000  | 0.82       |
| Random Forest     | 130,000  | 0.89       |
| XGBoost           | 126,000  | 0.90       |

### ✅ Best Model:
- **XGBoost** with the **lowest RMSE** and **highest R²**

---

## 📊 Next Steps

- Hyperparameter tuning (e.g. `GridSearchCV`)
- Deployment (e.g. Streamlit or Flask web app)
- Improve model via feature selection or model stacking

---

## 📑 Author

**Vincent Blanchard**  
GitHub: [VincentEquityLab](https://github.com/VincentEquityLab)  
Capstone Project – *Berkeley HAAS ML/AI Program*

---

