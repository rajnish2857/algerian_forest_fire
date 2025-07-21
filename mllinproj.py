import pickle
from flask import Flask,request,jsonify,render_template,redirect,url_for
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
ridge_model=pickle.load(open('model/ridge.pkl','rb'))
scaler_model=pickle.load(open('model/scaler.pkl','rb'))
@app.route("/")
def hello_world():
    return 'here you are'
@app.route("/predict",methods=['GET','POST'])
def prediction_fwi():
    if request.method=='POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        DC = float(request.form.get('DC'))
        ISI = float(request.form.get('ISI'))
        BUI = float(request.form.get('BUI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))
        new_data_scaled=scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)
        return render_template('home.html',results=result[0])
        
    
    return render_template('home.html')




if __name__== '__main__':
    app.run(host="0.0.0.0",debug=True)
