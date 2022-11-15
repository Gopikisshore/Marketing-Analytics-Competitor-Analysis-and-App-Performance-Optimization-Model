# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:23:55 2022

@author: Gopikishore S
"""

from flask import Flask, request, render_template, redirect, url_for
from analyzedata import generate_app_report
from Customizedstopwords import CustomizedstopwordsAndCreateWordCloud

app=Flask (__name__)

@app.route('/')
def index():
        return render_template('index.html')
        
@app.route('/result', methods=["POST","GET"])
def result():
    if request.method == "POST":
        keywords = request.form["keywords"]
        print(keywords)
        search_input = [ keyword.strip() for keyword in keywords.split(',')]
        No_of_review = int(request.form["No_of_ReviewS"])
        featuresIdentified = generate_app_report(search_input,No_of_review)
        print(featuresIdentified)
        return render_template('result.html',featuresToPrint=featuresIdentified)
        #return redirect(url_for('result.html'))
    else :
        return redirect(url_for('index'))

@app.route('/stopwords', methods=["POST","GET"])
def stopwords():
    if request.method == "POST":
        tokenCategory = request.form["SelectCategory"]
        stopwords = request.form["StopWords"]
        print(stopwords)
        stopwordsList = [stopword.strip() for stopword in stopwords.split(',')]
        NGram = request.form["No of words"]
        NofFeatures = request.form["No_of_features"]  
        featuresIdentified = CustomizedstopwordsAndCreateWordCloud(tokenCategory,stopwordsList,NGram, NofFeatures )
        print(featuresIdentified)
        resultHeading = NofFeatures + "with token category " + tokenCategory + " :"
        return render_template('stopwords.html',featuresToPrint=featuresIdentified,  resultHeading=resultHeading)
    else:
        initialValue = {'-':0}
        resultHeading = 'Click Get Result to view features:'
        return render_template('index.html',featuresToPrint=initialValue,  resultHeading=resultHeading)      

if __name__ == '__main__':
    app.run(debug=True)
