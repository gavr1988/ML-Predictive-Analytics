#trying to predict a certain flower belongs to based on information we have from our flowers

from sklearn.datasets import load_iris
#to reserve a selection of the data we will be using to train the model on and then test the model on the reserved data to see how well it performs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#clustering is a type of unsupervised learning where we group similar data points together. KMeans is a popular clustering algorithm that partitions the data into K clusters based on the mean of the data points in each cluster.
from sklearn.cluster import KMeans
#to check the accuracy of our model we will use the accuracy score function from sklearn.metrics which compares the predicted labels with the true labels and calculates the proportion of correct predictions.
from sklearn.metrics import accuracy_score
#install pandas to put data in data frame
import pandas as pd

#load the dummy data : iris dataset and create a data frame
def classification ():
    data = load_iris()
    X = data.data #represents attributes of the flowers e.g. petal length, width
    Y = data.target #Represents the category (0,1 or 2 represent species)
    
    #stage 2 is the splitting of our data 
    #X will be testing data
    #Y will be the training data
    #we are also reserving 20% of the data for testing and using a random state to ensure reproducibility of results

    X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.2, random_state = 42)
    
    #chosing algorithm to use for classification - we will use logistic regression

    model = LogisticRegression()

    #Stage 3: training the model
    model.fit (X_train, Y_train)

    predictions = model.predict (X_test)

    accuracy = accuracy_score (Y_test, predictions)
    print (f"Classification accuracy : {accuracy *100:.2f}%")




classification()
