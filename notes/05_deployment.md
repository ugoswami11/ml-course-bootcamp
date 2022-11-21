# Model Deployment

We have used jupyter notebook until now to build models but it's not meant for production environments because it may contain plots and other things that are useful for analysis or understanding of models but are not necessary in production. So for production we will extract our model into a pickle file and we will use webservices to generate predicitons.

We will be using the same churn prediction model that we created previuosly and we will deploy it using flask.

The below chart describes how we will converting our model which we created in jupyter notebook into a web service and how it will be used by the marketing serive to send mails to customers who are churning.

<img src="../notes/images/deployment.png">

- we will be creating a ```model.bin``` file in which we will store our model. 
- Then we will create a ```Churn webserive``` through which we will access our model and allow other components to access it and make predictions.
- The model then will be utilised by ```Marketing service```, the users input customer data into the service and the service communicates with Churn to request a prediction. Once the prediction is received, the service can execute whichever task is deemed appropiate, suchn as sending emails with offers to potentially churning customers.  

### Extracting the model

We will use pickle library to extract our model and store it into a binary file. we will create a python file or we will export our jupyter notebook as python file and we will store our data preparation, validation and trained model in it then we will export it via pickle to a bin file which we can use to generate prediction

We will also create a separate predict python file in which we wiil use our model created in previous step to generate prediciton. We will send the feature data via JSON file which will feed the model and the model will generate prediction based on that.

### Exposing the model as webservice

We will use Flask to expose our model as a webservice. we will use this framework in our predict.py file 