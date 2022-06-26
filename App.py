# Prem HG Predictor

from flask import Flask, redirect, render_template, request, session, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os
import base64
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sqlite3 as sl

app = Flask(__name__)
db = "eplmatchstats.csv"
conn = sl.connect(db, check_same_thread=False)

@app.route("/")
def home():  # go to log-in page using template
    if not session.get("logged-in"):
        return render_template("login.html", message="Welcome! Enter your favorite Premier League team from below:",
                               message1="Man United, Man City, Liverpool, Chelsea")


@app.route("/client/<user>")
def client(user):  # go to user.html
    if session["logged_in"] == False:  # if not logged in, redirect to login page
        return render_template("login.html", message="Welcome!, Enter your favorite Premier League team from below:",
                               message1="Man United, Man City, Liverpool, Chelsea")
    else:
        if user == "manutd":  # redirect to appropriate webpage depending on which club with graph of data points
            graph = create_graph("Man United")
            return render_template("manutd.html", club="Manchester United", plot_url=graph)

        elif user == "mancity":
            graph = create_graph("Man City")
            return render_template("mancity.html", club="Manchester City", plot_url=graph)
        elif user == "chelsea":
            graph = create_graph("Chelsea")
            return render_template("chelsea.html", club="Chelsea", plot_url=graph)
        elif user == "liverpool":
            graph = create_graph("Liverpool")
            return render_template("liverpool.html", club="Liverpool", plot_url=graph)

@app.route("/client/<user>/<year>/projection")
def client_projection(user, year):  # go to page to present prediction graph
    predgraph, prediction = create_projection(user, year)
    prediction = str(prediction)
    if user == "Man United":  # redirect to appropriate webpage depending on which club
        return render_template("manutdpred.html", club="Manchester United", YEAR=year, plot_url=predgraph, val=prediction)
    elif user == "Man City":
        return render_template("mancitypred.html", club="Manchester City", YEAR=year, plot_url=predgraph, val=prediction)
    elif user == "Chelsea":
        return render_template("chelseapred.html", club="Chelsea", YEAR=year, plot_url=predgraph, val=prediction)
    elif user == "Liverpool":
        return render_template("liverpoolpred.html", club="Liverpool", YEAR=year, plot_url=predgraph, val=prediction)


# show graph (history of club)
def create_graph(club):
    df = pd.read_csv("eplmatchstats.csv")
    fig, ax = plt.subplots(1, 1)
    df = df[df['HomeTeam'] == club]  # extrapolate data only for desired club
    df = df.dropna(subset='HomeGoals')  # drop Homegoals rows with NaN values
    df["Year"] = df["Date"].str[:4]  # create new column for years
    df_year = df.groupby("Year")
    year_list = df["Year"].unique()
    year_sums = []
    for year in year_list:  # find sum of each year
        df_group = df_year.get_group(year)
        year_sums.append(df_group["HomeGoals"].sum())
    # scatter plot of goals each year
    ax.scatter(year_list, year_sums, label="HomeGoals")
    ax.set(xlabel='YEAR', xticklabels=year_list, ylabel="Home Goals", title=club + " Home Goals")
    ax.legend()
    img = BytesIO()
    fig = ax.figure
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url

# create graph and Linear Regression for projection
def create_projection(club, year):
    print(year)
    df = pd.read_csv("eplmatchstats.csv")
    fig, ax = plt.subplots(1, 1)
    df = df[df['HomeTeam'] == club]
    df = df.dropna(subset='HomeGoals')  # drop Homegoals rows with NaN values
    df["Year"] = df["Date"].str[:4]  # create new column for years
    df_year = df.groupby("Year")
    year_list = df["Year"].unique()
    year_sums = []
    for years in year_list:  # find sum of homegoals in each year
        df_group = df_year.get_group(years)
        year_sums.append(df_group["HomeGoals"].sum())
    X_train, X_test, y_train, y_test = train_test_split(year_list, year_sums, test_size=0.25, random_state=0)
    regr = LinearRegression()
    regr.fit(X_train.reshape(-1,1), np.array(y_train).reshape(-1,1))
    X_test = np.array([year])
    y_pred = regr.predict(X_test.reshape(-1,1))  # predicted goal value for input 'year'
    ax.scatter(year_list, year_sums, label="HomeGoals")  # show plots of past stats
    ax.scatter(X_test, y_pred, color="black", label="Prediction")  # add plot of predicted value
    x_year_list = np.append(year_list, [year])
    ax.set(xlabel='YEAR', xticklabels=x_year_list, ylabel="Home Goals", title=club + " Home Goals")
    ax.legend()
    img = BytesIO()
    fig = ax.figure
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url, round(y_pred[0][0])


# after user has entered in predicted year, redirect to client page
@app.route("/action/createprojection", methods=["POST", "GET"])
def set_projection():
    if request.method == "POST":
        if request.form["year"].isdecimal():  # error check if input is valid year/all numbers
            return redirect(url_for('client_projection', user=session["club"], year=request.form["year"]))
        else:
            return render_template("login.html", message="Welcome! Enter your favorite Premier League team from below:",
                                   message1="Man United, Man City, Liverpool, Chelsea")

@app.route("/login", methods=["POST", "GET"])
def login():  # enter in which football club you want to see
    if request.method == "POST":  # post request
        if request.form["club"] == "Man United":
            session["club"] = request.form["club"]
            session["logged_in"] = True
            return redirect(url_for('client', user="manutd"))
        elif request.form["club"] == "Man City":
            session["club"] = request.form["club"]
            session["logged_in"] = True
            return redirect(url_for('client', user="mancity"))
        elif request.form["club"] == "Chelsea":
            session["club"] = request.form["club"]
            session["logged_in"] = True
            return redirect(url_for('client', user="chelsea"))
        elif request.form["club"] == "Liverpool":
            session["club"] = request.form["club"]
            session["logged_in"] = True
            return redirect(url_for('client', user="liverpool"))
        else:
            return render_template("login.html", message="Welcome! Enter your favorite Premier League team from below:",
                               message1="Man Utd, Man City, Liverpool, Chelsea")  # return to login page if wrong values


@app.route("/logout", methods=["POST", "GET"])
def logout():  # log out by modifying session values and redirect to home
    if request.method == 'POST':  # post request
        session['logged_in'] = False
        session.pop('username', None)
    return redirect(url_for('home'))

if __name__ == "__main__":
    
    app.secret_key = os.urandom(12)
    app.run(debug=True)


