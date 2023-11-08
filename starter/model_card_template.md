# Model Card

This model was implemented for the completion of the Udacity Machine Learning DevOps Engineer Nanodegree program, as part of the Project: Deploying a ML Model to Cloud Application Platform with FastAPI.

## Model Details
- Model Name: Random Forest Classifier
- Author: P. Arvanitis
- Date: 2023-11-08
- Framework/Library: scikit-learn

## Intended Use
The Random Forest Classifier model was developed to predict whether an individual's income exceeds $50K per year based on census data from the Adult dataset. The model can be used for various applications, including:

- Income prediction for individuals.
- Targeted marketing and advertising.
- Economic and social research.

## Training/Evaluation Data
The model was trained on a subset of the Adult dataset obtained from the UCI Machine Learning Repository. The training data was preprocessed and split into training and validation sets using a train-test split ratio of 80-20. The training data was used to train the model, while the validation data was used to evaluate the model's performance. The model was evaluated using the F1, precision, and recall metrics. 

- Training Data Source: https://archive.ics.uci.edu/dataset/20/census+income 
- Data Preprocessing: All spaces were removed from the data file. 
- Training Data Size: 26048 rows, 15 columns 
- Validation Data Size: 6513 rows, 15 columns

## Metrics
The model's performance was evaluated using the following metrics:
- Accuracy: 0.86 (training), 0.85 (validation)
- Precision: 0.82 (training), 0.79 (validation)
- Recall: 0.56 (training), 0.52 (validation)
- F1: 0.67 (training), 0.63 (validation)

From those metrics, the model's performance is considered acceptable. However, the generalization gap between the training and validation metrics indicates that the model may be overfitting the training data a little bit.

## Ethical Considerations
It is important to consider potential ethical and fairness implications when using or deploying this model. Some key considerations include:

- Bias: No specific bias mitigation techniques or fairness checks have been implemented. It is essential to carefully evaluate the model's predictions for any potential biases, especially concerning sensitive attributes such as race, gender, or age.
- Transparency: This model card is intended to provide transparency regarding the model's training data, intended use, and evaluation results.

## Caveats and Recommendations
- Hyperparameter Optimization: No hyperparameter optimization has been performed. It is advisable to explore hyperparameter tuning to potentially improve model performance.

- Model Updates: Consider periodic model updates and retraining as new data becomes available to maintain model accuracy.

- Bias Analysis: Conduct a detailed bias analysis, especially for sensitive attributes, and consider implementing bias mitigation strategies if necessary. A good starting point would be to observe the plots of the model's predictions for different subgroups of the data. These have been included in the 'slices_results' folder or can be generated using the 'run_slice_performance.py' script.

- Model Validation: Regularly validate the model's performance on new data to ensure it remains accurate and relevant.