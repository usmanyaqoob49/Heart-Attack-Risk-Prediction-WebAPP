from dash import Dash, dcc, html, Input, Output
import pandas as pd
import numpy as np

#importing visualizations from file
from visualization import age_categories_of_people_having_risk
from visualization import male_female_risk

#import function that will train model according to user selection
from visualization import train_model

#importing function for pre processing of the data and splitting
from visualization import pre_process_and_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve

#importing the file that has our data
df = pd.read_csv('heart.csv')

#calling the function to pre process and split data
X_train,  X_train, X_test, y_train, y_test = pre_process_and_split(df)


#initializing the Dash App
app = Dash(__name__)

#Frontend of the Application
app.layout = html.Div([
    html.H1('Heart Attack Prediction', style={'textAlign': 'center'}),
    html.H4('(This Web App uses algorithms that can predict if a person has the risk of a heart attack based on features.)',
            style={'textAlign': 'center'}),

    # Visualizations
    html.H3(html.B('Some Visualizations from the Dataset'), style={'textAlign': 'center'}),

    # Place both graphs in the same row using a flex container
    html.Div([
        html.Div([
            dcc.Graph(figure=age_categories_of_people_having_risk),
            html.H4('Number of People having Risk of a heart attack in different age categories.')
        ], style={'flex': 1, 'width': '50%'}),

        html.Div([
            dcc.Graph(figure=male_female_risk),
            html.H4('Number of Males and Females with Risk of Heart Attack.')
        ], style={'flex': 1, 'width': '50%'})
    ], style={'display': 'flex', 'flex-direction': 'row'}),

    # Model selection part
    html.Div([
        html.H2('Training Model on Data', style={'textAlign': 'center'}),
        html.H4('Select a model from the dropdown on which you want to train the data:', style={'textAlign': 'center'}),
        # Dropdown for model selection
        dcc.Dropdown(
            options=[
                {'label': 'Support Vector Classifier', 'value': 'Support Vector Classifier'},
                {'label': 'Logistic Regression', 'value': 'Logistic Regression'},
                {'label': 'Decision Tree Classifier', 'value': 'Decision Tree Classifier'},
                {'label': 'Random Forest Classifier', 'value': 'Random Forest Classifier'}
            ],
            value='Support Vector Classifier',  # Default selection
            id='models-dropdown',
            style={'textAlign': 'center'}
        )
    ]),

    # For showing results
    # Will receive output from train_model
    html.Div(html.B(id='accuracy-show'), style={'textAlign': 'center'}),

    #for line break
    html.Br(),html.Br(),html.Br(),
    #for real time prediction
    html.H2('Prediction on Your Data', style={'textAlign':'center'}),
    html.H4('Enter the Features Data:'),
    #place holders for user to enter features values
    html.Div([
        #taking value for the age feature
        "1- Give Age of the Person: ",
            dcc.Input(id= 'input-age',
                        value= 0,
                        type= 'number'),
        html.Br(), html.Br(),

        #taking value for the gender feature
        "2- Specify the Gender of the Person (1 for Male and 0 for Female): ",
            dcc.Input(id='input-gender',
                    value= 0,
                    type= 'number'),
         html.Br(), html.Br(),

        #taking value for the cp feature
        "3- Specify the Chest Pain Type (0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic): ",
            dcc.Input(id='input-cp',
                    value= 0,
                    type= 'number'),
         html.Br(), html.Br(),

        #taking value for the trtbps feature
        "4- Give Resting blood pressure (in mm Hg): ",
            dcc.Input(id='input-trtbps',
                    value= 0,
                    type= 'number'),
        html.Br(), html.Br(),

        #taking value for the chol feature
        "5- Give Cholestoral in mg/dl: ",
            dcc.Input(id='input-chol',
                    value= 0,
                    type= 'number'),
        html.Br(), html.Br(),

        #taking value for the fbs feature
        "6- Give fasting blood sugar > 120 mg/dl (1 for true, 0 for false): ",
            dcc.Input(id='input-fbs',
                    value= 0,
                    type= 'number'),
         html.Br(), html.Br(),

        #taking value for the restecg feature
        "7- Give Resting electrocardiographic results(0 = Normal, 1 = less severe level of abnormality, 2 = more severe or clinically significant abnormality.): ",
            dcc.Input(id='input-restecg',
                    value= 0,
                    type= 'number'),
        html.Br(), html.Br(), 


        #taking value for the thalachh feature
        "8- Give Maximum heart rate achieved: ",
            dcc.Input(id='input-thalach',
                    value= 0,
                    type= 'number'),
        html.Br(), html.Br(), 


        #taking value for the exng feature
        "9- Specify Exercis induced angina(1 = Yes:Higher risk of heart attacks, 0= No: Lower Risk of attack): ",
            dcc.Input(id='input-exng',
                    value= 0,
                    type= 'number'),
        html.Br(), html.Br(), 


        #taking value for the oldpeak feature
        "10- Specify Previous Peak Value: ",
            dcc.Input(id='input-oldpeak',
                    value= 0,
                    type= 'number'),
        html.Br(), html.Br(), 


        #taking value for the Slope feature
        "11- Specify Slope (Upsloping => slp = 0, Flat => slp = 1, Downsloping => slp = 2): ",
            dcc.Input(id='input-slp',
                    value= 0,
                    type= 'number'),
        html.Br(), html.Br(), 


        #taking value for the caa feature
        "12- Specify Number of major Vessels (A value in the caa column could range from 0 (no significant stenosis in major vessels) to 3 (significant stenosis in all three major vessels).): ",
            dcc.Input(id='input-caa',
                    value= 0,
                    type= 'number'),
        html.Br(), html.Br(), 


         #taking value for the thall feature
        "13- Specify Thalium Stress Test result(0=> Normal, 1=> Mild Abnormility, 2=>Moderate Abnormality, 3=>Severe Abnormality): ",
            dcc.Input(id='input-thall',
                    value= 0,
                    type= 'number'),
        html.Br(), html.Br(),

        #button for prediction
        html.Div([
            html.Button('Predict', id= 'predict-button', n_clicks=0)], style= {'textAlign':'center'}),
    
        html.Br(), html.Br(), html.Br(), html.Br(),


        
        #making div for output
        html.H2('Here is the Predicted Output for the data you have entered', style={'textAlign':'center'}),
        html.Div(
            #this id will be used in output of callback
            id= 'prediction-output'
        )
        
    ])
])

#-----------------------------callback to train data on the model of users choice--------------------------
@app.callback(
    # Output will be returned by the function train_model()
    Output('accuracy-show', 'children'),
    # Input will be our drop down id as we will take input from it
    Input('models-dropdown', 'value')
)
#value selected from drop down will be pass as input as we have given its id in Input
def train_selected_model(model_name):
        #passing the model name to training function imported from visualization.py
    #so from that function we will get back the instance of that specific model
    model= train_model(model_name)
    #now we need to train our data on that model
    model.fit(X_train, y_train)
    #now for prediction on the test data
    model_predictions= model.predict(X_test)
    #for the accuracy of model on test data
    model_accuracy = accuracy_score(model_predictions, y_test)
    return f'Accuracy of {model_name} is {model_accuracy}'
#-----------------------------------------------------------------------------------------------------------------

#-------------------------------callback for the prediction when user enters the values---------------------------
@app.callback(
    #output of callback will be the prediction shown
    Output(component_id= 'prediction-output', component_property='children'),

    #inputs will be all the inputs given by the user
    Input(component_id= 'input-age', component_property= 'value'),
    Input(component_id= 'input-gender', component_property= 'value'),
    Input(component_id= 'input-cp', component_property='value'),
    Input(component_id= 'input-trtbps', component_property= 'value'),
    Input(component_id= 'input-chol', component_property= 'value'),
    Input(component_id= 'input-fbs', component_property= 'value'),
    Input(component_id= 'input-restecg', component_property= 'value'),
    Input(component_id= 'input-thalach', component_property= 'value'),
    Input(component_id= 'input-exng', component_property= 'value'),
    Input(component_id= 'input-oldpeak', component_property= 'value'),
    Input(component_id= 'input-slp', component_property= 'value'),
    Input(component_id= 'input-caa', component_property= 'value'),
    Input(component_id= 'input-thall', component_property= 'value'),
    #input of the button
    Input(component_id= 'predict-button', component_property= 'n_clicks'),

    #also we will get input from model selection thing to check which model user has selected
    Input(component_id= 'models-dropdown', component_property='value')
)
#now function for the callback, we will pass all inputs and n_clicks (to ensure if predict button was predict)
def user_data_predict(age, gender, cp, trtbps, chol, fbs, restecg, thalach, exng, oldpeak, slp, caa, thall, n_clicks, selected_model):
    #putting them in a list
    features_data = pd.DataFrame({
            'age': [age],
            'sex': [gender],
            'cp': [cp],
            'trtbps': [trtbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalachh': [thalach],
            'exng': [exng],
            'oldpeak': [oldpeak],
            'slp': [slp],
            'caa': [caa],
            'thall': [thall]
        })    
    #now first of all select which model user has selected
    #this will give us instance of that model if it is logistic, random forest or anything
    model= train_model(selected_model)
    #train that model on our data first
    model.fit(X_train, y_train)

    #now we have to select conditions to check if user clicked predict or not
    #n_clicks will be zero if user has not pressed it
    if n_clicks!=0:
        #means user want to do prediction on the data
        prediction = model.predict(features_data)
        if prediction==1:
            return f'"According to Prediction of {selected_model} model, this person has the Risk of Heart Attack."'
        
        elif prediction==0:
            return f'"According to Prediction of {selected_model} model, this person does not have the Risk of Heart Attack."'
#-----------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
