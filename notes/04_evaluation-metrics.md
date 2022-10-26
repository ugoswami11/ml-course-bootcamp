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

We found out that the accuracy score is highest i.e. 80% when threshold is 0.5. we also look at the accuracy score when threshold is 1 and we got accuracy score around 73%.