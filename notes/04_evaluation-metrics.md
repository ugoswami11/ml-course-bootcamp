# Evaluation metrics for classification

In the classification notes we discussed on predicting the telco customer churn and we evaluated our model based on the accuracy score. So now we will look at different metrics that can be used to evaluate the model and will also check if accuracy score is correct metric in this case.

## Accuracy and Dummy model
Accuracy tells us about the fraction of correct prediction. 

Lets look at our churn prediction model, we predicted the probabilities of customers that will churn from the company. Now to calculate the accuracy we need to find the ratio between the correct prediction to the total number of customers
```
accuracy score = correct predictions/ total number of cusotmers(observations)
```

So we got around 80% accuracy on our predictions when we chose the threshold as 0.5 which means the any observation which got probability more than or equal to 0.5 we will mark them as churn and anything below that is not churn.

We now need to find our that the threshold we used is good or not. So for that we will take different threshold like 0, 0.3, 0.4, 0.5, 0.6, 0.7, 1 and we will generate our prediction based on that and calculate the accuracy score, whichever threshold will have the highest accuracy score we will use that threshold value in our final model.

We found out that the accuracy score is highest i.e. 80% when threshold is 0.5. we also look for the accuracy score when we use a dummy model i.e. threshold as 1 which means when no client will churn then we got accuracy score around 73%.

So when we compare our dummy model score with our original model prediction we don't see a huge difference which means the accuracy model is not the perfect measure of evaluation for this case.

## Confusion table
Confusion table is a way of measuring different types of errors and correct decisions that binary classifiers can make. Using this information, it is possible to evaluate the quality of the model by different strategies.

For binary classification, based on the prediction and the ground truth, there will be 4 posible outcome for our churn prediction scenario:

- When prediction is positive and outcome is correct- True positive
- When prediction is positive and outcome is incorrect- False positive
- When prediction is negative and outcome is incorrect- False negative
- When prediction is negative and the outcome is correct- True negative

The confusion table is a matrix whose columns (x dimension) are the predictions and the rows (y dimension) is the ground truth:
```
TP FP
FN TP
```
Each position contains the element count for each scenario. We can also convert the count values to percentages.

## Precision and Recall

Precision tell us the fraction of positive predictions that are correct. It takes into account only the positive class (TP and FP - second column of the confusion matrix), as is stated in the following formula:

Recall measures the fraction of correctly identified postive instances. It considers parts of the postive and negative classes (TP and FN - second row of confusion table). The formula of this metric is presented below:

In this problem, the precision and recall values were 67% and 54% respectively. So, these measures reflect some errors of our model that accuracy did not notice due to the class imbalance.