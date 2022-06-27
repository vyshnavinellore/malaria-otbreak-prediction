from flask import Flask,request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
df=pd.read_csv("content/sample_data/outbreak_detect.csv")
app = Flask(__name__)

#Deserialize
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html") #due to this function we are able to send our webpage to client(browser) - GET

@app.route('/predict',methods=['POST','GET'])  #gets inputs data from client(browser) to Flask Server - to give to ml model
def predict():
    features = [int(x) for x in request.form.values()]
    print(features)
    final = [np.array(features)]
    #our model was trained on Normalized(scaled) data
    X = df.iloc[:, 0:4].values
    sst=StandardScaler().fit(X)
    output = model.predict(sst.transform(final))
    print(output)

    if output[0]=='No':
        return render_template('index.html',pred=f'No')
    else:
        return render_template('index.html',pred=f'Yes')


if __name__ == '__main__':
    app.run(debug=True)