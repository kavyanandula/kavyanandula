Crime Data Analysis using Machine Learning Algorithms
Project Overview
This project analyzes a crime dataset from the UCI Machine Learning repository titled 'Crime and Communities'. The dataset consists of a large number of numerical columns, making it ideal for applying various machine learning algorithms. The goal of the analysis is to predict crime attributes such as Rape, Murder, Larceny, Robbery, Assault, Burglaries, Auto-theft, and Arsons by applying different machine learning models.

Dataset Description
The dataset contains information about:

Community names
County codes
Community codes
Urban population percentage
Age-based population
Gender-based population
Race-based population
These features help predict the occurrence of crimes. A binary label is created to predict whether a crime will occur based on selected features.

Enrichment Dataset
To enhance the dataset, we added a missing state column using the enrichment dataset (cities.json) containing mappings of latitude and longitude coordinates to respective cities and states. The enrichment dataset can be found here.

Machine Learning Algorithms Applied
Several machine learning algorithms were applied to the dataset to perform classification, regression, and clustering tasks. The algorithms include:

KMeans
GMM (Gaussian Mixture Model)
Linear Regression
Logistic Regression
Decision Trees
Naive Bayes
Random Forests
Support Vector Machine (SVM)
Principal Component Analysis (PCA)
K-Nearest Neighbors (KNN)
Analysis and Metrics
Each algorithm was evaluated based on its ability to predict crime attributes. The following metrics were used for evaluation:

Accuracy
F1 Score
Root Mean Squared (RMS) Value
Confusion Matrix
Conclusion
By applying the above machine learning algorithms, we were able to analyze the crime patterns effectively. The results from the classification, regression, and clustering models were used to form insights into crime occurrence across different communities, providing valuable information for crime prevention and resource allocation.
