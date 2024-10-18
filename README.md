# Machine Learning Projects

This repository contains various machine learning projects focusing on predictive modeling, deep learning, and risk assessment using diverse datasets and methods.

## Table of Contents

- [Project 1: Geolocation Prediction](#project-1-geolocation-prediction)
- [Project 2: Crime Analysis](#project-2-crime-analysis)
- [Project 3: Seismic Risk Assessment](#project-3-seismic-risk-assessment)
- [Project 4: EV Energy Optimization](#project-4-ev-energy-optimization)
-  [Project 5: Network Traffic Assignment using SiouxFalls Dataset](#project-5-Network-traffic-Assignment-using-SiouxFalls-Dataset)
- [Technologies Used](#technologies-used)



## Project 1: Geolocation Prediction

### Description
Developed a multimodal deep learning model for geolocation prediction based on Twitter texts and timestamps. This model achieved a Root Mean Squared Error (RMSE) of **10.615**, outperforming traditional unimodal methods.

### Key Features
- Utilizes both text data from tweets and temporal information for improved accuracy.
- Demonstrates the effectiveness of multimodal approaches in geographical predictions.

### Results
- RMSE: **10.615**
- Surpassed unimodal methods in performance.

## Project 2: Crime Data Analysis

### Description
This project analyzes crime data using machine learning algorithms from the UCI Machine Learning repository, titled **"Crime and Communities."** The dataset contains numerous numerical columns, making it suitable for machine learning applications without the need for extensive categorical data conversion.

### Dataset Overview
- **Source:** [Crime and Communities Dataset](https://www.kaggle.com/kkanda/analyzing-uci-crime-and-communities-dataset/data)
- **Features:** The dataset includes attributes such as Rape, Murder, Larceny, Robbery, Assault, Burglaries, Autotheft, and Arsons, along with community demographics like urban population percentage, age distribution, gender distribution, and racial distribution.

### Data Enrichment
- The state column was added using an enrichment dataset, **cities.json,** which maps latitude and longitude coordinates to their respective cities and states.

### Machine Learning Algorithms Applied
The following algorithms were utilized for various types of analyses based on the problem type (classification, regression, or clustering):
- KMeans
- Gaussian Mixture Model (GMM)
- Linear Regression
- Logistic Regression
- Decision Trees
- Naive Bayes
- Random Forest
- Support Vector Machine (SVM)
- Principal Component Analysis (PCA)
- K-Nearest Neighbors (KNN)

### Evaluation Metrics
We evaluated model performance using various metrics, including:
- Accuracy
- F1 Score
- Root Mean Square Error (RMSE)
- Confusion Matrix

### Conclusion
The analysis forms a comprehensive story that highlights key insights into crime occurrence and the effectiveness of different machine learning algorithms in predicting crime patterns.

### Key Features
- Implements machine learning techniques to classify and predict crime-related events.
- Achieved an accuracy of **86.86%** using the Random Forest model.

### Results
- Accuracy: **86.86%**

## Project 3: Seismic Risk Assessment

### Description
Working on a project related to machine learning-based seismic risk assessment using street view images. This project aims to analyze urban structures to evaluate potential risks associated with seismic events.

### Key Features
- Leverages street view images to gather data on building types and conditions.
- Utilizes machine learning techniques to assess risk and vulnerability.

## Project 4: EV Energy Optimization

### Description
The EV Energy Optimization project focuses on enhancing electric vehicle (EV) energy consumption at the household level. The objective is to improve energy efficiency and reduce costs associated with EV charging by analyzing household energy usage patterns and optimizing charging strategies using machine learning and time series analysis.

### Time Series Use Case
In this project, time series analysis is employed to understand and predict household energy consumption patterns over time. By utilizing historical energy usage data, the project aims to identify trends, seasonal variations, and anomalies in energy demand, which are crucial for optimizing EV charging strategies.

#### Key Objectives of Time Series Analysis:
- **Trend Analysis**: Identify long-term trends in household energy consumption and EV charging patterns to understand how energy usage evolves over time.
- **Seasonal Decomposition**: Analyze seasonal effects on energy consumption, such as increased usage during specific months or days (e.g., weekends, holidays) that could impact optimal charging times.
- **Anomaly Detection**: Detect unusual spikes or drops in energy consumption, which may indicate issues with charging infrastructure or abnormal household activity.
- **Forecasting**: Utilize historical data to forecast future energy consumption, helping to predict optimal times for EV charging based on anticipated energy demand.

### Data Sources
- Residential electric vehicle charging datasets from apartment buildings- Norway.


### Methodology
1. **Data Preprocessing**: Clean and preprocess the time series data to handle missing values, outliers, and ensure consistency in timestamps.
2. **Exploratory Data Analysis (EDA)**: Visualize energy consumption patterns to identify trends, seasonalities, and correlations with other factors (like temperature).
3. **Time Series Modeling**: Apply time series forecasting techniques (e.g., ARIMA, SARIMA, or Prophet) to predict future energy consumption.
4. **Optimization Strategy**: Develop machine learning models (e.g., Regression models, Decision Trees) to determine optimal charging times based on forecasted energy demand and cost.
5. **Validation**: Evaluate the accuracy of the forecasting models and the effectiveness of the optimization strategy using appropriate metrics (e.g., RMSE, MAE).


## Project 5: Network Traffic Assignment using SiouxFalls Dataset

### Description
This project involves implementing key traffic assignment algorithms—**Frank-Wolfe (FW) Algorithm** and **Method of Successive Averages (MSA)**—on the SiouxFalls network dataset. These algorithms are widely used in transportation engineering for finding equilibrium flow in traffic networks, optimizing routes, and minimizing overall travel time.

The **SiouxFalls dataset** represents a simplified model of a transportation network, making it ideal for testing and comparing traffic flow assignment algorithms.

#### Key Details:
- **Nodes**: Intersections or important locations in the transportation network.
- **Edges**: Roads connecting nodes, with attributes such as travel time or capacity.
- **OD Matrix**: Represents the demand (origin-destination pairs) for traffic flow between different nodes.

### Implemented Algorithms

#### 1. **Frank-Wolfe (FW) Algorithm**
The Frank-Wolfe algorithm is applied for solving the **traffic assignment problem**. This iterative algorithm calculates the user-equilibrium traffic flows, where no user can improve their travel time by switching routes. The objective is to minimize overall travel costs while accounting for congestion effects on travel time.

- **Input**: Network graph, travel time functions, demand matrix.
- **Output**: Flow distribution across all paths such that no user can unilaterally decrease their travel time (user equilibrium).

#### 2. **Method of Successive Averages (MSA)**
The MSA is a heuristic approach for **traffic assignment** that iteratively adjusts flows to converge to an equilibrium solution. It is often used in conjunction with other algorithms like Frank-Wolfe to ensure convergence to a stable traffic flow pattern in large-scale networks.

- **Input**: Initial traffic assignment, demand matrix.
- **Output**: Equilibrium flow distribution after successive iterations of flow averaging.


### Conclusion
This project demonstrates the use of traffic assignment algorithms for optimizing flow in transportation networks. By applying the Frank-Wolfe and Method of Successive Averages (MSA) algorithms to the SiouxFalls dataset, we achieved equilibrium flows, helping to minimize overall travel time in the network.



### Technologies Used
- **Python**: For data analysis and modeling.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical operations.
- **Matplotlib/Seaborn**: For data visualization.
- **Scikit-learn**: For implementing machine learning models.
- **Statsmodels**: For time series analysis and forecasting.






