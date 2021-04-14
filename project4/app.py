from flask import Flask, render_template, request, session, url_for, redirect
from flask_mysqldb import MySQL
import joblib

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
    if session:
        program = session['program']
        if program != 9:
         return render_template('profile.html', message=program)
        else:
            return render_template('loggedIn.html')
    else:
        message = []
        return render_template('login.html', message=message)

@app.route('/profile', methods=['GET'])
def profileRedirect():
    if session:
        cur = mysql.connection.cursor()
        print(session)
        username = session['username']
        cur.execute("Select * from userCredentials where username = %s", (username,))
        account = cur.fetchone()
        mysql.connection.commit()
        cur.close()
        print("Account below")
        print(account)
        program = account[3]
        print("Program below")
        print(program)
        if program == 9:
            return  render_template('loggedIn.html')
        else:
            return render_template('profile.html', message=program)
    else:
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
    if account:
        session['username'] = username
        session['program'] = account[3]
        session['loggedin'] = True
        session['id'] = account[0]
        print(account[0])
        return render_template('index.html')
    else:
        message = ["No user with these credentials found"]
        return render_template('login.html', message=message)

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    cur = mysql.connection.cursor()
    cur.execute("Select * from userCredentials where username = %s AND password = %s", (username, password))
    accountCheck = cur.fetchone()
    if accountCheck:
        errorMessage = 'An account with this username already exists'
        return render_template('login.html', message=[errorMessage])

    try:
        cur.execute("INSERT INTO userCredentials(username, password) VALUES (%s, %s)", (username, password))
    except:
        print("An error has occured")

    cur.execute("Select * from userCredentials where username = %s AND password = %s", (username, password))
    account = cur.fetchone()
    mysql.connection.commit()
    cur.close()
    print(account)
    if account:
        session['username'] = username
        session['program'] = account[3]
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
    if session:
        joblib_file = "productionModel.pkl"
        joblib_LR_model = joblib.load(joblib_file)

        joblib_LR_model
        print("-----------------------------------------------")
        testing_value = [[22, 0, 0, 0, 1]]
        score2 = joblib_LR_model.predict(testing_value)
        if 'loggedin' in session:
            # User is loggedin show them the home page
            print(session)
            # return render_template('loggedIn.html', username=session['username'])
            return render_template('loggedIn.html', username=score2)
        # User is not loggedin redirect to login page
        return redirect(url_for('login'))
    else:
        return render_template('login.html')

@app.route("/calculateFitness", methods=['POST'])
def calculateFitness():
    height = int(request.form["heightInput"])
    weight = int(request.form["weightInput"])
    age = int(request.form["weightInput"])
    children = int(request.form["childrenInput"])
    smoker = request.form["smokerInput"]
    sex = request.form['sexInput']
    if(smoker=="Yes"):
        smoker = int(1)
    else:
        smoker = int(0)
    if (sex == "Male"):
        sex = int(0)
    else:
        sex = int(1)
    bmi = calculateBMI(height, weight)
    joblib_file = "productionModel.pkl"
    joblib_LR_model = joblib.load(joblib_file)

    joblib_LR_model
    print("-----------------------------------------------")
    testing_value = [[age, sex, bmi, smoker, children]]
    fitness = joblib_LR_model.predict(testing_value)
    print(testing_value)
    print("----------------------------------------------")
    print(fitness)
    fitness = fitness[0]
    print(session['username'])
    cur = mysql.connection.cursor()
    username = session['username']


    cur.execute("Update userCredentials set program = %s where username = %s", (fitness, username))
    mysql.connection.commit()
    cur.close()
    print("Fitness below")
    print(fitness)
    return render_template('profile.html', message=fitness)

def calculateBMI(height, weight):
    height = height/100
    bmi = round((weight/(height * height)),1)
    return bmi

if __name__ == '__main__':
    app.run()
