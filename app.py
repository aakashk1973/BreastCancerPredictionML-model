import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

q = ""


@app.route("/")
def loadPage():
    return render_template('index.html', query="")


@app.route("/", methods=['POST'])
def predict():
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']

    model = pickle.load(open("modelss.sav", "rb"))

    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6,inputQuery7,inputQuery8,inputQuery9]]
    new_df = pd.DataFrame(data, columns=['Clump_Thickness', 'Uniformity_of_Cell_Size',
       'Uniformity_of_Cell_Shape', 'Marginal_Adhesion',
       'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
       'Normal_Nucleoli', 'Mitoses'])

    single = model.predict(new_df)
    probablity = model.predict_proba(new_df)[:, 1]

    if single == 1:
        o1 = "The patient is diagnosed with Breast Cancer"
        o2 = "Confidence: {}".format(probablity * 100)
    else:
        o1 = "The patient is not diagnosed with Breast Cancer"
        o2 = "Confidence: {}".format(probablity * 100)

    return render_template('index.html', output1=o1, output2=o2, query1=request.form['query1'],
                           query2=request.form['query2'], query3=request.form['query3'], query4=request.form['query4'],
                           query5=request.form['query5'])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)