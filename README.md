# Telecommunication-Customer-Churn
Classification/Prediction

Introduction: This project involves solving a binary classification machine learning problem in the Telecommunication industry.



Aim: The aim of this project is to make accurate predictions from historical data in regards to whether a Telecommunication customer will churn or not.



Methodology: While the fine details of the methodology followed in this project will be elaborated further in the relevant sections to come, it is to be mentioned that 16 machine learning models have been implemented, compared, and contrasted to identify the best performing one. Upon completion of the model development, the models have also been optimised to achieve higher accuracies. Although all models did not expeience an increase in predictive accuracy, the ones that have increased accuracy performance have achieved a 12% incrase. 

### 01. Import CPU Python Libraries
### 02. Function Helper
### 03. Import Dataset & Data Description
- Import CSV File
- Data Description
### 04. Data Understanding
- Data Information
- Data Summary Statistic
- Data Variance
### 05. Select the Featurs
### 06. Data Pre-Processing
- Drop Variables
- Convert Data Type
- Missing Value
### 07. Exploratory Data Analysis
- DV Visualization
- Categorical IDV
- Categorical IDV With DV
- Numerical IDV
- Numerical IDV With DV
### 08. Data Transformation
- Stander Scale
### 9. Feature Selection
- Wrapper - Forward
### 10. Feature Engineering
- LableEncoder
### 11. Statistics
- Correlations IDV with DV
- Correlation between all the Variables
### 12. Resampling Data
- SMOTE
### 13. Data Splitting
### 14. Standard Machine Learning Models
- Build the Models 'Train the Models'
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Histogram-based Gradient Boosting Classification Tree
  - AdaBoost Classifier
  - Extra Trees Classifier
  - K Neighbors Classifier
  - Naive Bayes Classifiers
  - Naive Bayes Classifier for Multivariate Bernoulli
  - Decision Tree Classifier
  - Logistic Regression Classifier
  - Logistic Regression CV Classifier
  - Stochastic Gradient Descent Classifier
  - Linear Perceptron Classifier
  - XGBoost Classifiers
  - Support Vector Machines Classifiers
  - Linear Support Vector Classification
  - Multilayer Perceptron Classifier
- Predication X_test
- Models Evaluation
  - Accuracy Score
  - Classification Report
  - Confusion Matrix
### 15. Optmization Machine Learning Models
- random grid for CPU Machine Learning Models
- Hyperparameters for CPU Machine Learning Models
- Build the Models 'Train the Models'
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Histogram-based Gradient Boosting Classification Tree
  - AdaBoost Classifier
  - Extra Trees Classifier
  - K Neighbors Classifier
  - Decision Tree Classifier
  - Logistic Regression Classifier
  - Logistic Regression CV Classifier
  - Stochastic Gradient Descent Classifier
  - Linear Perceptron Classifier
  - Support Vector Machines Classifiers
- Predication X_test
- Models Evaluation
  - Accuracy Score
  - Classification Report
  - Confusion Matrix

Results: For this particular classification problem, the highest performing model is that of Extra Trees Classifier which achieved an accuracy score of 93.8% with a F1 Score of xx, Recall of yy, and xyz score of zz.

![MicrosoftTeams-image (19)](https://user-images.githubusercontent.com/108016592/175177614-85041d7d-bff7-46af-967e-6cf4959fab76.png)

![MicrosoftTeams-image (20)](https://user-images.githubusercontent.com/108016592/175177648-793ac21f-1fb1-45fa-a603-835ab8cbccad.png)

Discussion: Although the Extra Tree Classifier has been observed to display the highest accuracy in comparison to all the other models, this score alone is not the sole determinant for choosing the Extra Tree Classifier model.

Upon closer inspection it is observed that the Extra Tree Classifier outperforms all the other models in F1 Score, Recall, and xyz. Should the latter scores have been observed to be higher in the other models, perhaps another model would have been determined to be the best performer as opposed to the Extra Tree Classifier.

![MicrosoftTeams-image (21)](https://user-images.githubusercontent.com/108016592/175178158-27b8d89d-e79a-4a01-96b1-6775b7c917c4.png)

![MicrosoftTeams-image (22)](https://user-images.githubusercontent.com/108016592/175178300-e2f26977-8c9d-4cbe-93e1-5719c31eb122.png)

![MicrosoftTeams-image (23)](https://user-images.githubusercontent.com/108016592/175178470-e6c60b3b-f3d3-42b8-878d-8b4264678354.png)


Conclusion:

The results achieved through most of the machine learning models in this project have achieved accuracy scores of above 80% and the best performer of all the 16 models that have been compared and contrasted has been determined to be the Extra Tree Classifier with an accuracy.



Having that said, it is to be mentioned that this performance can be expected to improve further by optimizing the models utilized. This can be done by experimenting further in terms of changing the hyper-parameters of the models, changing the method of feature selection, or using a different method of data transformation before deploying the machine learning models.
