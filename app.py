# import all required libraries for apply the ML model
from darts.models import NBEATSModel
from darts import TimeSeries
import pandas as pd
from flask import Flask, request, jsonify, render_template

# create an instance of Flask
app = Flask(__name__)

# load the model
model_loaded = NBEATSModel.load_model("model.pth.tar")


# this will be our homepage
@app.route("/")  # root page
def home():
    return render_template("index.html")  # render the template called index.html


# this my web API
# we provide some features to our ML model to return some outputs
@app.route("/predict", methods=["POST"])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # take all the values from the text fields in the html and store them in int_features variable
    int_features = [int(x) for x in request.from.values()]

    # change the values into pandas series
    sr_features = pd.Series(int_features)
    # change the dates into the right datatype

    # change the values into TimeSeries object
    train = TimeSeries.from_series(sr_features)

    # make the predictions
    pred_nbeat = model_loaded.predict(series=train, n=30)

    # change from timeseries to pandas series
    cashflow_serie = pred_nbeat.pd_series()

    # prediction text will get replaced with {{prediction_text}} in index.html
    return render_template("index.html", prediction_text="Cashflow predictions should be %s" % cashflow_serie)


# main function which run the whole flask
if __name__ == "__main__":
    app.run(debug=True)
