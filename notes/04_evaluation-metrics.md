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

We found out that the accuracy score is highest i.e. 80% when threshold is 0.5. we also look for the accuracy score when we use a dummy model i.e. threshold as 1 which means when no customer will churn then we got accuracy score around 73%.

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
Each position contains the element count for each scenario. We can also convert the count values to percentages and add up percentages for True negative and True positive to give us an idea for the accuracy of the model.

## Precision and Recall
Earlier we used accuracy as our metric which is the sum of true positive and true negative divided by the total number of predictions. In our churn prediction model it will the sum of prediction for cutomers that didnot churn and the customers that did churn divided by the total number of predictions. We can write this as below
```
accuracy= (TP + TN) / (TP + FP + FN + TN)
```
Precision tell us the fraction of positive predictions that are correct. In case of our churn prediction project it will only find out the fraction of customers that churned from the company predicted by the model.
The formula for precision is 
```
precision = TP / (TP + FP)
```
Recall measures the fraction of correctly identified postive instances. Instead of looking at all of the positive predictions, we look at the ground truth positives and we calculate the fraction of correctly identified positives in that subset It considers parts of the postive and negative classes (TP and FN). The formula of this metric is presented below:
```
recall = TP / (TP + FN)
```
In case of our churn predictions problem, the precision and recall values were 67% and 54% respectively. So, these measures reflect some errors of our model that accuracy did not notice due to the class imbalance.

## ROC

ROC stands for Receiver Operating Characteristic, and this idea was applied during the Second World War for evaluating the strength of radio detectors. This measure considers two factors False Positive Rate (FPR) and True Postive Rate (TPR), which are derived from the values of the confusion matrix.

FPR is the fraction of false positives (FP) divided by the total number of negatives (FP and TN ), and we want to minimize the FPR value. The formula of FPR is the following:
```
FPR = FP / ( TN + FP )
```

In the other hand, TPR or Recall is the fraction of true positives (TP) divided by the total number of positives (FN and TP - second row of confusion table), and we want to maximize the TPR value. The formula of this measure is presented below:
```
TPR = TP / (TP + FN)
```

If we try different thresholds and calculate confusion tables for each threshold, we can also calculate the TPR and FPR for each threshold.

When we plot the FPR (x axis) against the TPR (y axis), a random baseline model should describe an ascending straight diagonal line, a perfect model would increase inmediately to 1 and stay up, and our model most likely will be somewhere in between in a bow shape, ascending quickly at first and then decreasing the growth until it reaches the point (1,1).

A good model would be a very "arched" bow, as close as possible to the perfect model and as far away as possible form the diagonal.

## ROC AUC
The ROC AUC is the Area under the ROC curves which can tell us how good is our model with a single value. For a random model with a diagonal ROC curve (worse scenario), the ROC AUC will be 0.5. For a perfect model with a perfect ROC curve that instantly rises to 1 (best scenario), the ROC AUC will be 1.0.

In ther words, AUC can be interpreted as the probability that a randomly selected positive example has a greater score than a randomly selected negative example.

The AUC is actually the probability of a random positive sample having a higher score than a random negative sample.

## K-Fold Cross Validation

K-fold Cross Validation consists on evaluating the same model on different subsets of data.In this algorithm, the full training dataset is divided into k partitions, we train the model in k-1 partiions of this dataset and evaluate it on the remaining subset. Then, we end up evaluating the model in all the k folds, and we calculate the average evaluation metric for all the folds. 

We can then compute the AUC score for each permutation and then calculate the mean and standard deviation of all of them to get the average prediction and the spread within predictions.

This method is applied in the parameter tuning step, which is the process of selecting the best parameter.

