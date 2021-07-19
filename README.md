# GRIP-TheSparkFoundation-Internship July 2021
# TASK 1: Prediction using Supervised ML
This is a simple linear regression task as it involves just 2 variables.

# Softwares and Libraries

This task uses the following softwares and libraries:
* Google Colab
* Python
* numpy
* pandas
* matplotlib
* seaborn
* scikit-learn

# Machine Learning Models used

* In this task I have implemented a Basic Linear Regression Model from scratch with gradient descent as an optimization technique and RSquared(r2score)  and Root Mean Squared(RMSE) error as evaluation metrics.
* LinearRegression Model from scikit-learn library.

# Description
The Objective of this task id to predict the score of students given the number of hours of study.
The datset provided has just 2 features 
* 1.Hours (No of hours of study) - Independent Variable
* 2.Scores (Marks secured by students) - Dependent Variable(Target Variable).

Data Analysis was performed on the data and the following observations were noted:
* Data Distribution is fairly Symmetrical - Based on the value of skewness and distribution plots of seaborn.
* Target Variable is in linear relationship with independent variable.
Dataset was split into train set and test set respectively with test_size=0.2.
Linear Regression Model implemented from scratch was trained on the training set and the model performance was evaluated on the following evaluation metrics:
* Mean Absolute Error
* Mean squared Error
* Root Mean Squared Error
To acheive better performance Linear Regression model from Scikit_learn Library was trained on the train set and its performance is evaluated based on the metrics given above.
To the conclusion, The predicted score of student given 9.5 hrs of study was found to be 93.68.

# Task2 : Prediction using Unsupervised ML
From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.
This is a classification task, where we need to classify the iris dataset into its species
* Iris-setosa
* Iris-versicolor and
* Iris-virginica 
respectively and visualize the clusters and their centroids.

# Softwares and Libraries
This task uses the following softwares and libraries:
* Google Colab
* Python
* numpy
* pandas
* matplotlib
* seaborn
* scikit-learn

# Machine Learning Models used:
* KMeans Clustering from scikit-learn library.

# Description:
The dataset has 4 Independent variables and 1 Dependent variable:
* SepalLengthCm
* SepalWidthCm
* PetalLengthCm
* PetalWidthCm 
* Species   

Exploaratory Data analysis was performed on the Iris Dataset and the following important observations were noted:
* The Data Distribution is fairly Symmetrical.
* Features Petal Width and Petal Length makes fair distinction among the Species.
* Except for SepalWidthCm rest of colums are free of outliers.

The rows with outliers were removed from the dataset.
Using The Elbow method the optimum number of clusters were found which is n_clusters = 3 beyond which KMean's model's performance has not imporved much.
For purpose of visualization the categorical column Species has been transformed into numerical column.
Keeping n_clusters=3 the KMeans model was trained and respective True Vs KMeans clusters on different combinations were plotted.

- You can find complete code and details in .ipynb notebook.
