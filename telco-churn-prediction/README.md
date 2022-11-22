## Telco Churn Prediction

The Telco Customer Churn Prediction project is developed as part of my learnings from the [Data talks club ml zoomcamp](https://datatalks.club/courses/2021-winter-ml-zoomcamp.html) organized by [Alexey Grigorev](https://twitter.com/Al_Grigor). The project is about predicting the cutomer will churn or not based on the features provided in the dataset using pandas, numpy and sci-kit learn library.

### Dataset
The dataset is taken from Kaggle and is available at this [link](https://www.kaggle.com/blastchar/telco-customer-churn)

### Project development notebooks
The process of data preparation, model development and evaluation is available on the below links
- [Data Preparation and Model development](https://github.com/ugoswami11/ml-course-bootcamp/blob/main/notebooks/telco-customer-churn.ipynb)
- [Evaluation metrics](https://github.com/ugoswami11/ml-course-bootcamp/blob/main/notebooks/evaluation-metrics.ipynb)

### How to run?

- Install all the dependencies using pipenv to run the model
``` 
Pipenv install
```
- Run the flask webservice by executing the below command inside Pipenv
```
python predict.py
```
- Generate prediction by executing below command inside pipenv
```
python predict-test.py
```

