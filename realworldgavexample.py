#trying to predict the performance of a student based on their academic data and attendance, demographic etc.
 import load_studenthistoricaldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import pandas as pd

#load the student data
def predicted_student_performance():
    data = student_data()
    X = data.data #represents the student data, e.g. attendance, demographic, key stage 2 scores, test data
    Y = data.target #Represents the potential outcome for the student based on the student data and students who have been in similar positions in the past. 
    
    #stage 2 is the splitting of our data 
    #X will be testing data
    #Y will be the training data
    #we are also reserving 20% of the data for testing and using a random state to ensure reproducibility of results

    X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.2, random_state = 42)
    
    #chosing algorithm to use for this  - we will use logistic regression

    model = LogisticRegression()

    #Stage 3: training the model
    model.fit (X_train, Y_train)

    predictions = model.predict (X_test)

    accuracy = accuracy_score (Y_test, predictions)
    print (f"student predicted grade accuracy : {accuracy *100:.2f}%")




predicted_student_performance()