from flask import Flask, render_template, request, session, url_for, redirect
from flask_mysqldb import MySQL
import os

app = Flask(__name__)

app.config
app.config['MYSQL_HOST'] = 'firstdatabase.cdclx9ozp7gb.eu-west-1.rds.amazonaws.com'
app.config['MYSQL_USER'] = 'admin'
app.config['MYSQL_PASSWORD'] = 'ProjectPassword'
app.config['MYSQL_DB'] = 'demoData'

app.secret_key = 'abcd'

mysql = MySQL(app)
@app.route('/')
def home():
    cur = mysql.connection.cursor()
    username = "tester"
    password = "password"
    cur.execute("INSERT INTO userCredentials(username, password) VALUES (%s, %s)", (username, password))
    mysql.connection.commit()
    cur.close()
    return render_template('home.html')

@app.route('/loginRedirect', methods=['GET'])
def loginRedirect():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    cur = mysql.connection.cursor()
    cur.execute("Select * from userCredentials where username = %s AND password = %s", (username, password))
    account = cur.fetchone()
    mysql.connection.commit()
    cur.close()
    print(account)
    if account:
        session['username'] = username
        session['loggedin'] = True
        session['id'] = account[0]
        print(account[0])
    return render_template('index.html')

@app.route('/pythonlogin/home')
def returnHome():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('home.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
@app.route('/testfunction')
def testfunction():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        print(session)
        return render_template('loggedIn.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
if __name__ == '__main__':
    app.run()
