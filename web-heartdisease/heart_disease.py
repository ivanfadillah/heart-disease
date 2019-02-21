import os
import sqlite3
from flask import Flask , request , session , g , redirect , url_for , abort , \
     render_template , flash
from flask import send_from_directory
import pandas as pd
import numpy as np
import itertools
import math
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

app = Flask(__name__)

@app.route("/inputform")
def show_form():
    return render_template("Inputform.html")

@app.route("/")
def show_homepage():
    return render_template("HomePage.html")

@app.route('/result' , methods=['POST','GET'])
def show_result():
    if request.method == 'POST':

     return "POST"
    else:
        name = request.args.get('name')
        age = '?' if (request.args.get('age')=='') else int(request.args.get('age'))
        sex = '?' if (request.args.get('sex')==None) else int(request.args.get('sex'))
        chestpaint = '?' if (request.args.get('chest-paint')=='') else int(request.args.get('chest-paint'))
        restingblood = '?' if (request.args.get('resting-blood')=='') else request.args.get('resting-blood')
        cholestrol = '?' if (request.args.get('cholestrol')=='') else request.args.get('cholestrol')
        fastingblood = '?' if (request.args.get('fastingblood') == '') else request.args.get('fastingblood')
        restingecg = '?' if (request.args.get('restingecg')=='') else int(request.args.get('restingecg'))
        maxheart = '?' if(request.args.get('max-heart')=='') else request.args.get('max-heart' ) 
        exerciseInc = '?' if (request.args.get('exerciseInc')==None) else request.args.get('exerciseInc') 
        stdepression = '?' if (request.args.get('stdepression')=='') else request.args.get('stdepression')
        peakexer = '?' if (request.args.get('peakexer')==None) else request.args.get('peakexer')
        numberofmajor = '?' if (request.args.get('numberofmajor' )=='') else request.args.get('numberofmajor')
        thal = '?' if (request.args.get('thal')==None) else request.args.get('thal') 
       # return maxheart    
        #diagnosis of heart disease/ angiographic disease status
        file_test = "static/tubes2_HeartDisease_test.csv"
        df_test = pd.read_csv(file_test)

        feature_test = df_test

        test_data = pd.DataFrame({'Column1':[age] , 'Column2':[sex] , 'Column3':[chestpaint],
        'Column4':[str(restingblood)] , 'Column5':[str(cholestrol)] , 'Column6':[str(fastingblood)] , 
        'Column7':[restingecg] , 'Column8':[str(maxheart)] , 'Column9':[str(exerciseInc)],
        'Column10':[str(stdepression)] , 'Column11':[str(peakexer)] , 'Column12':[str(numberofmajor)] ,
        'Column13':[str(thal)]}, index=[len(feature_test)])

        frames = [feature_test, test_data]
        test_input = pd.concat(frames)
        
        header = test_input.columns.values.tolist()
        feature_impute_test = test_input.replace('?',np.nan)

        imputer_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

        discrete_value = ['Column1','Column2','Column3','Column6','Column7','Column9','Column13']
        continues_value = ['Column4','Column5','Column8','Column10','Column11','Column12']
        best_header_mlp = ['Column4', 'Column5', 'Column6', 'Column11', 'Column12']

        imputer_mode.fit(feature_impute_test[discrete_value])
        feature_impute_test[discrete_value] = imputer_mode.transform(feature_impute_test[discrete_value])

        imputer_mean.fit(feature_impute_test[continues_value])
        feature_impute_test[continues_value] = imputer_mean.transform(feature_impute_test[continues_value])

        feature_impute_test['Column13'] = pd.to_numeric(feature_impute_test['Column13'])


        feature_scale_test1 = pd.DataFrame(preprocessing.scale(feature_impute_test), columns=header)

        MLP = joblib.load('static/best_model_mlp.pkl')
        result_message = MLP.predict(feature_scale_test1[best_header_mlp])

       # return "%s dan %s" %(str(result[len(feature_test)]),str(result[len(feature_test)-1]))
       # return "%s %s %s %s %s %s %s %s %s %s %s %s %s" %(str(age), str(sex), str(chestpaint), str(restingblood),
       # str(cholestrol), str(fastingblood), str(restingecg), str(maxheart), str(exerciseInc), str(stdepression),
       # str(peakexer), str(numberofmajor), str(thal) )

        test_data1 = pd.DataFrame({'Column1':[age] , 'Column2':[sex] , 'Column3':[chestpaint],
        'Column4':[str(restingblood)] , 'Column5':[str(cholestrol)] , 'Column6':[str(fastingblood)]}, index=[len(feature_test)])
        
        
        test_data2  = pd.DataFrame({'Column7':[restingecg] , 'Column8':[str(maxheart)] , 'Column9':[str(exerciseInc)],
        'Column10':[str(stdepression)] , 'Column11':[str(peakexer)] , 'Column12':[str(numberofmajor)] ,
        'Column13':[str(thal)]}, index=[len(feature_test)])       

                
        result = [str(name),result_message[len(feature_test)], test_data1, test_data2]
        return render_template('ResultPage.html' , result=result)

