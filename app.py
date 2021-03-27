from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("home.html",tables=[df.to_html(classes='data')], titles=df.columns.values)

@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
    	return "Helloooo"
    if request.method == 'POST':
        form_data = request.form
        return form_data

