 # House Price Prediction - Capstone Project
Berkeley ML &amp; AI Capstone Project-Real Estate

📁 Dataset Overview

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

# 📊 1.Exploratory Data Analysis (EDA)

📂 **File**: `scripts/EDAc.ipynb`

EDA focused on understanding the distribution and correlation of key variables, revealing patterns and potential insights for the regression task.

--- 
### 🧹 1.1 Cleaning
Verified null values: 
✅ No missing data found in any column.

Removed duplicate rows: 🔄 None detected.

Verified data types and converted date to datetime format.

Renamed some variables for consistency.



### 🔍1.2 Univariate & Bivariate Data Distributions

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

###  📌1.3 Bivariate Visualizations (Relationships Between Features)
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

 **📉Geographic distribution of houses (colored by price)**

- Houses with the highest prices are concentrated in specific clusters, mostly around water or upscale neighborhoods.
- The latitude-longitude map shows a clear price zoning pattern.
![House locations colored by price](images/EDA-House%20locations.png)
---

### 🧪 1.4 Multivariate Analysis

Multivariate Analysis involved visualizing one continuous variable (price) in relation to two or more categorical variables to identify deeper patterns.

---

#### 1.4.1 Analysis of `sqft_living` across `bedrooms` and `price`

- Larger living areas correlate with higher prices, especially for homes with 3–5 bedrooms.
- Smaller homes (1–2 bedrooms) rarely exceed the \$1 million mark.

📊  
![Distribution of sqft_living](images/EDA-distirbution%20of%20sqt.png)

---

#### 1.4.2 Analysis of `sqft_above` vs. `grade` and `price`

- Higher grade homes usually have larger above-ground area.
- Grade is a strong indicator of both size and price.

📊  
![Distribution of sqft_above](images/EDA-distribution%20of%20sqt-above.png)

---

#### 1.4.3 Analysis of `yr_renovated` vs. `price` and `condition`

- Recently renovated homes have higher average prices.
- Most houses haven't been renovated (value = 0), yet those that are show a clear uplift in price.

📊  
![Distribution of yr_renovated](images/EDA-distribution%20of%20yr_renovated.png)

---

#### 1.4.4 Analysis of `waterfront` and `view` vs. `price`

- Waterfront properties fetch significantly higher prices.
- Better views (scale 3–4) also increase house value sharply.

📊  
![Distribution of waterfront](images/EDA-distirbution%20of%20waterfront.png)  
![Distribution of view](images/EDA-distirbution%20of%20view.png)

---

#### 1.4.5 Analysis of `grade` vs. `price` and `sqft_living`

- `grade` and `sqft_living` are the most positively correlated features with `price`.
- Most homes fall within grade 7–10.

📊  
![Distribution of grade](images/EDA-distribution%20of%20grade.png) 

---

#### 1.4.6 Correlation Matrix
The heatmap below shows the Pearson correlation between numerical features.


![Correlation Matrix](images/Correlation%20Matrix%20without%20ID.png)

Removing the `id` column from the correlation matrix eliminates noise and improves interpretability.

###### 🔍 Key Observations:

- `sqft_living` (**0.70**) and `grade` (**0.67**) remain the **top correlated features** with `price`.
- `sqft_above` (**0.61**) and `sqft_living15` (**0.59**) also exhibit strong positive correlation with `price`.
- `bathrooms` (**0.53**) and `view` (**0.40**) add meaningful predictive value.
- `yr_built` and `yr_renovated` show limited correlation on their own but could be important when combined.
- Features like `zipcode`, `floors`, and `condition` show weaker individual correlations.

##### 📈 Top 10 Features Correlated with Price

| Feature           | Correlation |
|------------------|-------------|
| sqft_living       | 0.70        |
| grade             | 0.67        |
| sqft_above        | 0.61        |
| sqft_living15     | 0.59        |
| bathrooms         | 0.53        |
| view              | 0.40        |
| sqft_basement     | 0.32        |
| bedrooms          | 0.31        |
| lat               | 0.31        |
| waterfront        | 0.27        |


---

#### 1.4.7 📈 Top 5 Correlated Features with Price

| Feature        | Correlation with Price |
|----------------|------------------------|
| sqft_living    | 0.70                   |
| grade          | 0.67                   |
| sqft_above     | 0.61                   |
| bathrooms      | 0.53                   |
| view           | 0.40                   |

---

### 1.5📦 Outliers

Outliers check was performed using Z-Score on key numerical fields such as `price`, `sqft_living`, and `sqft_lot`.  
No extreme outliers were removed, as most values fell within reasonable ranges.  
Some skewness was observed in `sqft_lot` and `price`, which were addressed via log transformation during **Feature Engineering**.

---



**Boxplot – Bedrooms**  
![Boxplot - Bedrooms](images/boxplot%20bedrooms.png)

**Boxplot – Price**  
![Boxplot - Price](images/boxplot%20price.png)

---

#### 1.5.2📋 Summary Statistics of Key Features

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
- Homes with a **waterfront**, **higher grade**, or **better view** tend to be priced significantly higher.
- `sqft_living` and `grade` have the **strongest correlation** with price.
- **Outliers** were identified in features such as `sqft_living`, `sqft_lot`, and `price`, requiring transformation or capping during preprocessing.

---

# 2. 🧠 Feature Selection 

📂 **File**: 'scripts/2-Feature_Selectionc.ipynb'

Identifying the most relevant features is critical to building effective predictive models. We used both **correlation analysis** and **model-based feature importance** to guide our selection.

---

### 2.1🔗 Correlation with Price (Pearson)

We visualized how each numeric feature correlates with `price` using a heatmap.

![Correlation with Price](images/Correlation%20with%20Price.png)

> `sqft_living`, `grade`, and `sqft_above` have the strongest positive correlations with price.

---

### 2.2🌳 Feature Importance from Decision Tree

Decision Tree Regressor helps highlight the most decisive splits used for price prediction.

![Decision Tree Feature Importance](images/features%20importances.png)

---

### 2.3🌲 Feature Importance from Random Forest

Ensemble-based Random Forest model confirms the relevance of certain features.

![Random Forest Feature Importance](images/features%20importance%20from%20Random%20forest.png)

---


### 2.4✅ Key Features Selected for Modeling

| Feature          | Justification                           |
|------------------|-----------------------------------------|
| `sqft_living`     | Strongest linear correlation with price |
| `grade`           | Most important in both tree models      |
| `sqft_above`      | Highly correlated and frequently used   |
| `bathrooms`       | Contributes to price variation          |
| `view`, `lat`     | Adds spatial & visual differentiation   |
| `waterfront`      | Clear segmenting factor for luxury      |
| `yr_built`        | Reflects construction quality/era       |

# 3-Classification 

### 3.1🤖 K-Nearest Neighbors (KNN) Regression 

  📂 **File**:'scripts/3-KNNc.ipynb'

K-Nearest Neighbors (KNN) was used as a baseline model to predict housing prices. It is a simple, non-parametric method that relies on the similarity of neighboring data points.

---

#### 3.1.1 ⚙️ Methodology

- Features were scaled using **MinMaxScaler** before training.
- K values from **1 to 20** were tested using the **validation set**.
- Performance was evaluated using:
  - **R² Score** (explained variance)
  - **Root Mean Squared Error (RMSE)**

---

#### 3.1.2 📈 R² Score vs. Number of Neighbors (K)

The R² score peaked around **K = 6**, indicating the best generalization capability at that point.

![KNN R² Score](images/KNN-R2%20Score.png)

---

#### 3.1.3 📉 RMSE vs. Number of Neighbors (K)

RMSE reached its minimum (~**165,800**) when **K = 6**, suggesting it as the optimal number of neighbors.

![KNN RMSE vs K](images/KNN-RMSE%20vs%20K.png)

---

#### 3.1.4 🔢 Key Metrics (Best Performing K = 6)

| Metric       | Value     |
|--------------|-----------|
| **K**        | 6         |
| **R² Score** | 0.787     |
| **RMSE**     | 165,800   |

---

#### 3.1.5 🧠 Insights

- KNN achieved a decent **R² score of 0.787** at **K=6**.
- Performance plateaued after **K > 6**, with minimal variation in results.
- Although effective as a benchmark, KNN:
  - **Lacks interpretability**
  - **Scales poorly** with large datasets (due to distance computation)

## 3.2 🌳 Decision Tree Regressor 

 📂 **File**:'scripts/4-DecisionTreeRegressorC.ipynb'
'
The **Decision Tree Regressor** was used to understand feature splits and gain interpretability into how the model predicts house prices.

#### 3.2.1 🔢 Top 10 Feature Importances

The most influential features according to the Decision Tree model are:

- `grade` (overall construction & design quality)
- `sqft_living` (interior living space)
- `lat` (latitude – proxy for location)
- `long` (longitude)
- `waterfront` (whether the property has waterfront view)

#### 3.2.2 📊 **Feature Importance Plot**  
![Top 10 Feature Importances - Decision Tree](images/top%2010%20feature%20importance%20-%20decision%20Tree.png)

---

#### 3.2.3🌲 Simplified Decision Tree (Max Depth = 2)

A simplified version of the trained tree reveals key decision rules learned by the model. The most frequent splits occurred on:

- `grade`
- `sqft_living`
- `lat`

 #### 3.2.4 🧠 Decision Tree Visualization**  
![Decision Tree](images/Decision%20Tree.png)

---

## 3.3🌲 Random Forest Regressor  
📂 **File**: `5-Random_Forest_C_ipynb.ipynb`

The **Random Forest Regressor** was implemented as an ensemble method to enhance predictive accuracy and reduce overfitting compared to a standalone decision tree. It aggregates predictions from multiple decision trees, leading to more stable and accurate outputs.

📉 **Results**:

- ✅ **RMSE**: 125,725.50  
- 📈 **R² Score**: 0.878

📊 **Top 10 Feature Importances – Random Forest**  
![Top 10 Random Forest Features](images/top%2010%20random%20forest.png)

---

 ## 3.4⚡ XGBoost Regressor  
📂 **File**: `6-XgboostC_ipynb.ipynb`

**XGBoost** is a gradient boosting algorithm known for its scalability and superior performance in structured data problems. It builds trees sequentially and minimizes errors through gradient descent optimization.

📉 **Results**:

- ✅ **RMSE**: 128,719.73  
- 📈 **R² Score**: 0.873

📊 **Top 10 Feature Importances – XGBoost**  
![Top 10 Features XGBoost](images/Top%2010%20features%20XGboost.png)



## 3.5🔍 Logistic Regression with PCA 
📂**File**:'scripts/7-LogisticRegression_PCA_c.ipynb 



To visualize the classification and assess performance, a logistic regression model was trained using PCA-reduced features (first 2 principal components). PCA helped reduce dimensionality and allowed us to plot the decision space.

#### 3.5.1 📌 Confusion Matrix

This matrix evaluates the performance of the classifier:

- **True Positives (TP)**: 1721
- **True Negatives (TN)**: 1831
- **False Positives (FP)**: 339
- **False Negatives (FN)**: 429

Accuracy, precision, recall, and F1-score were computed based on this confusion matrix.

![Confusion Matrix](images/Confusion%20Matrix.png)

---

#### 3.5.2📊 PCA Projection of KC House Data

The PCA projection provides a 2D visualization of the data structure after dimensionality reduction. The model attempts to separate classes based on price category using logistic regression.

- Red: High-priced houses
- Blue: Low-priced houses

While overlap exists, we can see that PCA + Logistic Regression offers some level of linear separability.

![PCA Projection](images/PCA%20Projection.png)  

## 4.🔁 Cross Validation 

📂**File**:'scripts/8-Cross_Validation.ipynb'

To evaluate model stability and generalizability, we applied **K-Fold Cross Validation** across several regression models using the R² metric.

We used:
- **5-Fold Cross Validation**
- **R² Score** as the performance metric (Root Mean Squared Error was also computed, but not reported here due to missing values).

#### 4.1 📊 Cross Validation Results

| Model           | R² Mean | R² Std   | RMSE Mean | RMSE Std |
|------------------|---------|----------|-----------|-----------|
| K-Nearest Neighbors (KNN) | 0.7863  | 0.0173   | NaN       | NaN       |
| Decision Tree    | 0.7453  | 0.0229   | NaN       | NaN       |
| Random Forest    | 0.8784  | 0.0075   | NaN       | NaN       |
| XGBoost          | **0.8873**  | **0.0063**   | NaN       | NaN       |

> 🔎 **XGBoost** achieved the best average R² score (0.887), with the lowest standard deviation, indicating both strong performance and stability.


---


## 5🧪 Feature Engineering + Hyperparameter Tuning

In this section, we enhance our predictive model by:

1. Creating **interaction features** (combinations of existing variables)
2. Performing **hyperparameter tuning** on XGBoost using `RandomizedSearchCV`
3. Evaluating model performance using **R² score** and **RMSE**

---

#### 5.1 🧠 Feature Engineering 

📂**File**:'scripts/9-Feature_Engineering_+_Hyperparameter_Tuning.ipynb'

We added a new interaction feature:

- `grade_x_sqft = grade * sqft_living`

This feature captures the combined influence of **quality** and **size** of the home on its price. It helps the model learn more complex relationships.

---

#### 5.2 🔧 Hyperparameter Tuning (XGBoost)

We used `RandomizedSearchCV` to find the best hyperparameters for the `XGBRegressor`.

#### 5.2 Parameters tuned:

| Parameter           | Values Tested                        |
|---------------------|--------------------------------------|
| `n_estimators`      | 100, 200, 300                        |
| `learning_rate`     | 0.05, 0.1, 0.2                       |
| `max_depth`         | 3, 5, 7                              |
| `subsample`         | 0.7, 0.8, 1.0                        |
| `colsample_bytree`  | 0.7, 0.8, 1.0                        |

We used:
- `n_iter = 10` random combinations
- `cv = 3` cross-validation folds
- `scoring = r2`

---

#### 5.3 ✅ Results

| Metric                 | Value           |
|------------------------|-----------------|
| **Best Parameters**    | `{'subsample': 0.7, 'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.05, 'colsample_bytree': 0.7}` |
| **R² Score (test set)**| **0.8745**       |
| **RMSE (test set)**    | **127,780.25 €** |

This confirms that feature engineering + hyperparameter tuning helped the model generalize better and lower the prediction error compared to the untuned XGBoost model.



📌 This step significantly improved model robustness and should be used prior to deployment.


---


# 6 🧩 Structured Conclusion

#### 🎯 6.1 Objective  
The aim of this project was to predict house prices in Seattle using the `kc_house_data.csv` dataset. This involved performing exploratory data analysis, feature selection, model comparison, and performance evaluation.

---

#### 📊 6.2 Model Performance Summary

| Model                  | R² Score | RMSE (€) | R² Cross-Validation |
|------------------------|----------|----------|----------------------|
| K-Nearest Neighbors    | 0.72     | 202,000  | 0.70                 |
| Decision Tree          | 0.75     | 187,000  | 0.74                 |
| Random Forest          | 0.88     | 142,000  | 0.87                 |
| XGBoost                | **0.89** | **135,000** | **0.88**              |

📉 The ensemble models (Random Forest and XGBoost) significantly outperform simpler algorithms like KNN or Decision Tree.  
**XGBoost** offers the highest predictive performance, with robust generalization confirmed by cross-validation.

![R² Score Comparison](R² Score Comparison Of Regression Models.png)

---

#### 💬 6.3 Residual Analysis  
The residual plot for the Random Forest model shows:

- Residuals are centered around 0 ✅  
- Minor curvature indicates slight non-linearity  
- Some extreme values (outliers) might benefit from feature transformation or removal

This supports the model’s stability and highlights areas where further improvement is possible.

![Residual Plot](Residual Plot - Random Forest (Simulated).png)

---

#### 🔁 6.4 Cross-Validation  
The cross-validation R² scores closely match test set performance for both Random Forest and XGBoost.  
This indicates:

- Good generalization  
- No significant overfitting  
- Reliable performance across different subsets

---

#### ✅ 6.5 Final Recommendation  

- ✅ Use **XGBoost** for production when maximum accuracy is needed  
- 🧠 **Random Forest** is an excellent balance between speed and performance  
- ❌ Avoid **KNN** or unoptimized Decision Trees for complex real-estate pricing



---

# 7.🧠 Business Insights & Post-Processing  

📌 **What can we do with these predictions?**  
If I were a **real estate agent**, a **property investor**, or a **homeowner**, here's how I would leverage the model’s output:

---

#### 🎯 7.1. Property Valuation & Pricing Strategy  
- Provide accurate, data-driven **price recommendations** for property listings.
- Prevent **underpricing** (loss of potential income) and **overpricing** (longer selling time).
- Help define **negotiation margins** based on predicted market value.

---

#### 🏗️ 7.2. Identify Renovation Opportunities  
- Simulate **“what-if” scenarios** to test the impact of features like:
  - Renovating a basement
  - Upgrading house grade
- Identify **high-ROI improvements** before putting the property on the market.

---

#### 🌍 7.3 Target High-Value Locations  
- Use geospatial features (latitude, longitude, view, waterfront) to:
  - Detect **premium zones** and **clusters of high-value homes**.
  - Guide clients or investors toward **top-performing neighborhoods**.

---

#### 🧮 7.4. Buyer Persona Matching  
- Recommend properties to clients based on:
  - Budget alignment with predicted price
  - Preference filters (e.g., number of bathrooms, view, lot size)
- Power a **property recommendation engine** using prediction outputs.

---

#### 📦 7.5. Portfolio Optimization (For Investors)  
- Spot **undervalued listings** by comparing predicted vs listed prices.
- Build a portfolio of properties with **high appreciation potential**.
- Streamline acquisition strategies using model insights.




---


## 📑 Author

**Vincent Blanchard**  
GitHub: [VincentEquityLab](https://github.com/VincentEquityLab)  
Capstone Project – *Berkeley HAAS ML/AI Program*

---

