<div align="center">

 # 🚗 Car Dheko - Used Car Price Prediction

</div>

<div align="center">

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)
[![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=google-cloud&logoColor=white)](https://www.google.com/)

</div>


##  📊 Project Overview

The "Car Dheko - Used Car Price Prediction" project aims to develop an accurate machine learning model that predicts the prices of used cars. The project also involves deploying this model as an interactive web application using Streamlit. This tool will be user-friendly and accessible to both customers and sales representatives, allowing them to input car features and obtain real-time price predictions.

<div align="center">

<img src="https://github.com/user-attachments/assets/730d7010-4da9-44b4-8f92-15278d36761b" width="800" />

<img src="https://github.com/user-attachments/assets/af742224-abf8-47d6-9fc7-046ceed2b2cd" width="800"/>

</div>

## 🔍 Project Scope

The project leverages historical data on used car prices from CarDekho, which includes various features such as make, model, year, fuel type, transmission type, and more. The objective is to create a machine learning model that can accurately predict used car prices based on these features. The model will be integrated into a Streamlit-based web application, which will allow users to input car details and receive price estimates instantly.

## 🎯 Problem Statement

Enhancing customer experience and streamlining the pricing process at Car Dheko by leveraging machine learning. The goal is to create a user-friendly tool that predicts the prices of used cars based on various features and deploy it as an interactive web application.

## 🔗 Data Source

* We would be working with quite a large data which contains about __8369__ data points where again we would be dividing that into the training set and the test set.
* Having a look at some of the cars that we are always excited to use in our daily lives, it is better to understand how these cars are being sold and their average prices.
* Feel free to take a look at the dataset that was used in the process of predicting the prices of cars. Below is the link.

 __Source:__ https://www.cardekho.com/usedCars

## 📂 Dataset

The dataset comprises multiple Excel files, each representing data from a different city. It includes detailed information about used cars listed on CarDekho, such as their specifications and available features. The data needs to be preprocessed to handle missing values, standardize formats, encode categorical variables, and normalize numerical features.

Feature Description Link : [Features](https://docs.google.com/document/d/1hxW7IvCX5806H0IsG2Zg9WnVIpr2ZPueB4AElMTokGs/edit)

## 📈 Metrics

Predicting used car prices is a __continuous machine learning problem__. Therefore, the following metrics that are useful for regression problems are taken into account. Below are the __metrics__ that was used in the process of predicting car prices.

* [__Mean Squared Error (MSE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
* [__Mean Absolute Error (MAE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
* [__R-squared (R2)__](https://www.investopedia.com/terms/r/r-squared.asp)
* [__Root Mean Squared Error (RMSE)__](https://c3.ai/glossary/data-science/root-mean-square-error-rmse/)

## 🚀 Approach
 
### 1. Data Processing

* Import and concatenate the datasets for various cities, convert them into a structured format, and concatenate them into a single dataset.

**Data before structuring (unstructured data)**
 
<img width="955" alt="dictionary" src="https://github.com/user-attachments/assets/ad1529fb-5f02-4c7c-9f97-10cdf89ca4df">

* Address missing data using imputation techniques—mean, median, or mode for numerical columns, and mode or a new category for categorical columns.

* Next is standardizing the Data. Ensure that the data is in the correct format, e.g., removing units from numerical columns and converting them to integers.

* Encoding categorical variables using one-hot encoding for nominal categorical variables and label or ordinal encoding for ordinal categorical variables.

* Normalizing numerical features using techniques like Min-Max Scaling or Standard Scaling.

* Identify and remove or cap outliers using methods such as the IQR (Interquartile Range) method or Z-score analysis.

**Data after structuring**

<div align="center">

<img width="850" src="https://github.com/user-attachments/assets/b5fdcbca-7a39-4af3-8640-cdf3c62ed8eb">

<img width="850" src= "https://github.com/user-attachments/assets/cbbfa850-b2c7-48a5-91a7-b22d6d2a27cb">

</div>

### 2. Exploratory Data Analysis (EDA)

In this section, the dataset is thoroughly explored to uncover patterns, trends, and insights that can guide further analysis and model development. EDA helps in understanding the data better and in identifying any anomalies, relationships, or key features that may influence the model's performance.

#### 2.1 Statistics Calculated:

Provide a quantitative overview of the dataset by calculating key summary statistics.

**Mean:** The average value for each numerical feature, providing insight into the central tendency.
 
**Median:** The middle value when the data is ordered, useful for understanding the distribution, especially in the presence of outliers.
 
**Mode:** The most frequently occurring value, particularly relevant for categorical variables.
 
**Standard Deviation:** Measures the dispersion of the data points from the mean, indicating how spread out the values are.

#### 2.2 Data Visualization

Visualizing data is crucial for identifying patterns, relationships, and outliers that may not be immediately apparent from summary statistics alone. The following

visualizations were created:

**(i) Scatter Plots :**

Visualize relationships between two numerical variables. For example, a scatter plot of price vs. mileage can reveal how the price decreases as mileage increases.

<img width="999" alt="image" src="https://github.com/user-attachments/assets/aa3c3cd5-f5f0-4895-8a07-7c60a9e38017">

**(ii) Histograms :**

Show the distribution of a single numerical variable. For instance, a histogram of engine_size can help identify the most common engine sizes in the dataset.

<img width="999" alt="image" src="https://github.com/user-attachments/assets/e99a8464-617f-4e9b-80f8-554ee4ade2d9">

**(iii) Box plot :**

The plot_boxplots function generates box plots for specified columns in a DataFrame to help identify outliers and understand the distribution of the data. Box plots are useful for visualizing data distributions and detecting any anomalies or extreme values.

<img width="999" alt="image" src="https://github.com/user-attachments/assets/c09d7b26-4b13-4a59-9f72-e5aa7265f2ec">

**(iv) Correlation heatmap :**

The plot_correlation_heatmap_fix function generates a correlation heatmap for numeric features in a DataFrame, providing a visual representation of feature relationships. This helps in identifying correlations and understanding how features relate to each other.

<img width="999" alt="image" src="https://github.com/user-attachments/assets/95936b40-c53d-4f86-a457-2f5242106b51">

#### 2.3 Insights and Observations

From the summary statistics and visualizations, several interesting patterns and trends were observed:

**Insight 1:** A negative correlation between mileage and price, indicating that cars with higher mileage tend to have lower prices.

**Insight 2:** Certain engine sizes are more common, suggesting they may be more desirable or widely available in the market.

**Insight 3:** Outliers in the price distribution indicate the presence of luxury or high-end vehicles that could skew the analysis.

#### 2.4 Feature Selection

Feature selection is a critical step in the modeling process that involves identifying the most important features that significantly impact car prices. By selecting the most relevant features, the model becomes more efficient, interpretable, and performs better.

**2.4.1 Techniques Used for Feature Selection**

To identify the important features that have a significant impact on car prices, the following techniques were employed:

**(i) Correlation Analysis:**

* Objective: Assess the linear relationship between numerical features and the target variable (car prices).
 
* Method: Calculate the correlation coefficients between the features and the target variable. Features with high positive or negative correlation are considered important.

**(ii) Feature Importance from Models:**

* Objective: Leverage machine learning models to determine which features contribute most to the model’s predictions.
 
* Method: Use algorithms like Random Forest or Gradient Boosting, which can provide a measure of feature importance as part of their training process.

**(iii) Domain Knowledge:**

* Objective: Use industry-specific insights and expert knowledge to identify features that are logically expected to influence car prices.

* Method: Incorporate knowledge of factors like brand reputation, engine specifications, and market trends that are known to affect car valuation.

* Example: High-end brands or cars with advanced safety features are generally expected to have higher prices. These insights guide the inclusion or exclusion of certain features in the final model.

<img width="999" alt="image" src="https://github.com/user-attachments/assets/c605f2c4-7a23-4374-b520-da6936b457ff">

By focusing on these selected features, the model is likely to achieve better performance, with improved accuracy and interpretability, while reducing the risk of overfitting by removing irrelevant or redundant features.

### 3. Model Development

* Divide the dataset into training and testing sets to evaluate the model's performance on unseen data.

* Commonly, the dataset is split into 70% training data and 30% testing data, or 80% training and 20% testing. This ensures that the model is trained on a substantial portion of the data while retaining enough data for a reliable test.

* Choose appropriate machine learning algorithms.

**(i) Algorithms:**

   **- Linear Regression:** Useful for modeling the relationship between the dependent and independent variables when the relationship is linear.
 
   **- Decision Trees:** Helpful for capturing complex, non-linear relationships in the data.
 
   **- Random Forests:** An ensemble method that combines multiple decision trees to improve predictive performance and control overfitting.
 
   **- Gradient Boosting Machines:** Another ensemble method that builds models sequentially to correct errors made by previous models.

* Ensure the model's performance is consistent and not dependent on a particular training subset.Use cross-validation techniques such as k-fold cross-validation to evaluate the model on different subsets of the data.

* Enhance the model's performance by tuning its hyperparameters.

 **(ii) Methods:**
 
  **- Grid Search :** Exhaustively searches through a specified parameter grid to find the best combination.
   
  **- Random Search :** Randomly samples parameters from a distribution to find a good combination more efficiently

### 4. Model Evaluation

**4.1 Performance Metrics:**

To assess how well your machine learning models are performing, you'll use several key metrics:

**(i) Mean Absolute Error (MAE):**

  Measures the average magnitude of errors in a set of predictions, without considering their direction. It’s the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.

**(ii) Mean Squared Error (MSE):**

  Measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value. This metric penalizes larger errors more than MAE.

**(iii) R-squared (R²):**

  Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, where a higher value indicates a better fit.

**(iv) Root Mean Squared Error (MSE):**

  Root Mean Squared Error (RMSE) is a commonly used metric to evaluate the accuracy of a regression model. It measures the square root of the average of the squared differences between the predicted values and the actual values. RMSE gives more weight to larger errors because the errors are squared before averaging, making it sensitive to outliers.

#### 4.2 Model Comparison

To choose the best model, you should compare the models you have trained based on the performance metrics mentioned above. Here’s how you can approach this:

* Calculate Metrics: Compute MAE, MSE, and R² for each model you have trained.
 
* Compare Scores: Evaluate which model has the lowest MAE and MSE and the highest R².
 
* Select the Best Model: Choose the model that best balances these metrics according to your specific needs and objectives.

|  __Machine Learning Models__  |  __Mean Absolute Error (MAE)__  | __Root Mean Squared Error (RMSE)__ | __Mean Squared Error (MSE)__ | __R-squared (R2)__ |
| :-: | :-: | :-: | :-: | :-: |
| [__1.&nbsp;Linear&nbsp;Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) |  4.38171  |  6.30828  | 3.57944 | 0.72963 |
| [__2.&nbsp;Decision&nbsp;Tree&nbsp;Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) |  2.70174  |  5.11750  | 2.61888 | 0.82207 |
| [__3.&nbsp;Gradient&nbsp;Boosting&nbsp;Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) | 2.90385 | 4.76576 | 2.27124 | 0.84568 |
| [__4.&nbsp;Random&nbsp;Forests&nbsp;Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) | __2.15680__ | __4.08725__ | __1.65056__ | __0.88650__ |

* Random Forest has the lowest MSE, the lowest RMSE , and the highest R² .This suggests that it has the best overall performance, providing the most accurate predictions and the best fit to the data.
 
### 5. Optimization

#### 5.1 Feature Engineering

Feature engineering involves creating new features or modifying existing ones to enhance the model’s performance. This process often involves:

  **(i) Creating New Features:**
 
  * Derive new features from existing data that might better capture the underlying patterns.

  **(ii) Modifying Existing Features:**

   * Transform existing features to improve their contribution to the model. Like Normalizing or scaling features to bring them onto a similar scale,Encoding categorical variables appropriately, and Handling missing values through imputation or feature engineering.

 #### 5.2 Regularization

Regularization techniques are used to prevent overfitting by adding a penalty to the model's complexity. Key techniques include:

  **5.2.1 Lasso Regression (L1 Regularization):**

  Adds a penalty equal to the absolute value of the magnitude of coefficients. This can lead to sparse models where some feature weights are exactly zero, effectively performing feature selection.  

  **5.2.2 Ridge Regression (L2 Regularization):**

  Adds a penalty equal to the square of the magnitude of coefficients. This helps in reducing the complexity of the model and can handle multicollinearity.

  **5.2.3 Choosing the Right Regularization Technique:**

  Depending on the problem and the dataset, you might choose Lasso if feature selection is important, or Ridge if you need to handle multicollinearity and reduce overfitting.

By applying these optimization techniques, you can improve the performance and generalization of your model, leading to better predictive accuracy and robustness.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/b10a8cfc-c098-4b68-b22f-a0fbf470d235">

### 6. Streamlit Application:

* This section describes how the final model was deployed as an interactive web application using Streamlit. The application allows users to input various car features and receive real-time predictions for the car's price.
 
#### 6.1 Streamlit Setup

* Installation: Ensure Streamlit is installed in your environment.

```
 pip install streamlit

```
* Application Structure: The Streamlit app is structured to have an intuitive user interface where users can input car details such as make, model, year, mileage, engine size, and more.

#### 6.2 Building the User Interface

* Input Fields: Streamlit provides various input widgets like sliders, text boxes, and dropdowns. These allow users to specify the car features they want to input for the price prediction.

* Prediction Button: A button that, when clicked, triggers the model to make a prediction based on the input features

#### 6.3 Integrating the Model

* The trained machine learning model is loaded and used to make predictions based on the user inputs.

* The model processes the input data and returns a price prediction, which is then displayed in the app.

#### 6.4 Running the Application

* Running Locally: You can run the application locally using the following command:

```
streamlit run app.py
```

* Deploying the Application: You can deploy the Streamlit app using platforms like Streamlit Cloud, Heroku, or any other hosting service that supports Python web applications

<img width="959" alt="image" src="https://github.com/user-attachments/assets/2ff0e0e5-90f6-4705-a86e-d091ec5b161e">

## 📜 Conclusion

**1.Accurate Predictive Model for Used Car Prices**

Developed a robust machine learning model capable of predicting used car prices with high accuracy. The model demonstrates strong performance metrics, including low Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), ensuring reliable price predictions.

**2.In-Depth Data Analysis and Visualizations**

Conducted a thorough exploratory data analysis (EDA) that included generating summary statistics and creating various visualizations such as scatter plots, histograms, box plots, and correlation heatmaps. These analyses provided valuable insights into the dataset, revealing key patterns and relationships that influenced the model’s performance.

**3.Comprehensive Documentation**

Provided detailed documentation that outlines the methodology used in the project, including data preprocessing, feature engineering, model development, and evaluation. The documentation also includes explanations of the different machine learning models tested and their performance results, ensuring clarity and transparency.

**4.Interactive Streamlit Application**

Developed an interactive Streamlit application that allows users to input car features and obtain real-time price predictions. This user-friendly application makes the model accessible to non-technical users and facilitates easy and immediate use of the predictive tool.
