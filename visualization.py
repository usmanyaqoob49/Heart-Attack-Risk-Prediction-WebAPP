import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
# Scaling
from sklearn.preprocessing import RobustScaler

# Train Test Split
from sklearn.model_selection import train_test_split

# Models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Metrics
from sklearn.metrics import accuracy_score, classification_report, roc_curve

df = pd.read_csv('heart.csv')

#people who have risk of heart attack
risk_class= df[df['output'] == 1]

top_10= risk_class['age'].value_counts().reset_index()
top_10.columns = ['Age Category', 'Count']
top_10= top_10[0:10]
top_10.sort_values(by= 'Count', ascending= True)

#visualization 1
age_categories_of_people_having_risk = px.bar(data_frame= top_10,
                                              x= 'Age Category',
                                              y= 'Count')
age_categories_of_people_having_risk.update_layout(
    plot_bgcolor='gray',  # Set the background color of the plot area to gray
    paper_bgcolor='black'  # Background color of the entire graph
)
age_categories_of_people_having_risk.update_traces(marker_color='white')


male_female_count= risk_class['sex'].value_counts().reset_index()
male_female_count.columns= ['Sex', 'Number of Individual']

male_female_count['Sex'].replace(1, 'Male', inplace= True)
male_female_count['Sex'].replace(0, 'Female', inplace= True)


#visualization 2

male_female_risk = px.bar(data_frame= male_female_count,
                                              x= 'Sex',
                                              y= 'Number of Individual')
male_female_risk.update_layout(
    plot_bgcolor='gray',  # Set the background color of the plot area to gray
    paper_bgcolor='black'  # Background color of the entire graph
)
male_female_risk.update_traces(marker_color='white')

def pre_process_and_split(data):
    # creating a copy of df
    df1 = data
    # encoding the categorical columns
    # making the features and labels ready for model training
    #everything except output
    features= df1.drop(['output'],axis=1)
    #label
    labels = df1['output']
    # instantiating the scaler
    scaler = RobustScaler()
    # scaling the continuous featuree
    X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size = 0.2, random_state = 42)
    return X_train,  X_train, X_test, y_train, y_test

#calling the above function to get data
X_train,  X_train, X_test, y_train, y_test = pre_process_and_split(df)

def train_model(model_name):
    #if user has selected the svc from drop down 
    if model_name == 'Support Vector Classifier':
        #SVC
        clf = SVC(kernel='linear', C=1, random_state=42)

        # predicting the values
        # y_pred = clf.predict(X_test)
        # svc_accuracy = accuracy_score(y_test, y_pred)
        return clf
    
    #if user has selected logistic regression
    elif model_name =='Logistic Regression':
        lr = LogisticRegression()
        # lr.fit(X_train, y_train)
        # y_pred_proba = lr.predict_proba(X_test)
        # y_pred = np.argmax(y_pred_proba,axis=1)
        # logistic_reg_accuracy = accuracy_score(y_test, y_pred)
        return lr


    #if user has selected decision tree classifier
    elif model_name=='Decision Tree Classifier':
# instantiating the object
        dt = DecisionTreeClassifier(random_state = 42)
        # fitting the model
        # dt.fit(X_train, y_train)
        # calculating the predictions
        # decision_tree_y_pred = dt.predict(X_test)
        # decision_tree_accuracy= accuracy_score(y_test, decision_tree_y_pred)
        return dt
    
    #if user has selected random forest
    elif model_name=='Random Forest Classifier':
        # instantiating the object
        rf = RandomForestClassifier()
        # fitting the model
        # rf.fit(X_train, y_train)
        # calculating the predictions
        # y_pred = rf.predict(X_test)
        # random_forest_accuracy= accuracy_score(y_test, y_pred)
        return rf



