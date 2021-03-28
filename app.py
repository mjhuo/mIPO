from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import datetime
from datetime import date
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from GoogleNews import GoogleNews

app = Flask(__name__)
data = pd.read_csv("cleaned_ipo_data.csv")

daysBeatSP = data.iloc[:, 2].values

priceChangeAfter100 = data.iloc[:, 4].values

feature_variables = data.drop(['daysBeatSP', 'priceChange100Days'], axis=1)

sector_dummies = pd.get_dummies(feature_variables["sector"], prefix = "sector", drop_first = True)
feature_variables = pd.concat([feature_variables, sector_dummies], axis=1)
feature_variables = feature_variables.drop(['sector'], axis=1)

class predictors():

	def passArguments(a):
		price = a['Price']
		if(price == ""):
			price = int(15)
		ipo_date = a['Date']
		if(ipo_date == ""):
			spvalue = 3974.54 #Current S&P 500 value
		if(ipo_date != ""):
			sp = ipo_date.split('-')
			entered_date = date(int(sp[0]), int(sp[1]), int(sp[2]))
			first_date = date(2021, 3, 26)
			diff = int((entered_date - first_date).days)
			spvalue = 3974.54*((1.005)**diff)
		which_sec = str(a['sector'])
		if(which_sec == ""):
			which_sec = "Miscellaneous"
		sector = "sector_"+which_sec
		argument_df = pd.DataFrame(columns=feature_variables.columns.values)

		listForThis = []
		columns = []
		for i in feature_variables.columns.values:
		    columns.append(i)
		    if(i == "initialPrice"):
		        listForThis.append(float(price))
		    if(i == "spvalue"):
		        listForThis.append(float(spvalue))
		    if(i == sector):
		        listForThis.append(int(1))
		    if(i != sector and i != 'spvalue' and i != 'initialPrice'):
		        listForThis.append(int(0))
		a_series = pd.Series(listForThis, index = feature_variables.columns)
		argument_df = argument_df.append(a_series, ignore_index=True)

		return argument_df

	def predict_SP(arguments):
		regressor_SP = RandomForestRegressor(n_estimators=500, random_state=0)
		regressor_SP.fit(feature_variables, daysBeatSP)
		SP_prediction = regressor_SP.predict(arguments)

		return str(round(SP_prediction[0]))

	def predict_100(arguments):
		regressor_100 = RandomForestRegressor(n_estimators=500, random_state=0)
		regressor_100.fit(feature_variables, priceChangeAfter100)
		prediction_100 = (regressor_100.predict(arguments)+1)*100

		return str(round(prediction_100[0], 2))

	def sentiment(term): # Used https://github.com/krishnaik06/Stock-Sentiment-Analysis/blob/master/Stock%20Sentiment%20Analysis.ipynb
		df=pd.read_csv('news_data.csv', encoding = "ISO-8859-1")
		train = df
		data=train.iloc[:,2:27]
		data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
		list1= [i for i in range(25)]
		new_Index=[str(i) for i in list1]
		data.columns= new_Index
		for index in new_Index:
		    data[index]=data[index].str.lower()
		' '.join(str(x) for x in data.iloc[1,0:25])

		headlines = []
		for row in range(0,len(data.index)):
		    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))
		countvector=CountVectorizer(ngram_range=(2,2))
		traindataset=countvector.fit_transform(headlines)
		randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
		randomclassifier.fit(traindataset,train['Label'])
		test_transform= []

		#Google News: https://medium.com/analytics-vidhya/googlenews-api-live-news-from-google-news-using-python-b50272f0a8f0

		googlenews = GoogleNews()
		googlenews.search(term)
		result=googlenews.result()
		news_df=pd.DataFrame(result)
		test_dataset = countvector.transform(news_df)
		predictions = randomclassifier.predict(test_dataset)
		return predictions[0]

@app.route('/')
def hello_world():
    return render_template("home.html")

@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
    	return "Helloooo"
    if request.method == 'POST':
        form_data = request.form
        term = form_data['Name']
        sentiment = (predictors.sentiment(term))
        s = ""
        if(sentiment == 1):
        	s = "According to the latest news, the sentiment around " + term + " is positive!"
        else:
        	s = "According to the latest news, the sentiment around " + term + " is negative..."

        arguments = predictors.passArguments(form_data)
        prediction_SP_fin = predictors.predict_SP(arguments)
        prediction_100_fin = predictors.predict_100(arguments)

        pred_SP_string = "Our model predicts that in it's first year of trading, it will outperform the S&P 500 about " + prediction_SP_fin + " days (out of ~250)."
        pred_100_string = "Meanwhile, our model also predicts that after 100 days, the stock will be valued be " + prediction_100_fin + "% of the initial price."
        report_string = []
        report_string.append(s)
        report_string.append(pred_SP_string)
        report_string.append(pred_100_string)

        return render_template("report.html", text = report_string)
