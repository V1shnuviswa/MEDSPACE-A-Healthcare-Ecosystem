from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/find_hospital')
def find_hospital():
    return "Find Hospitals Page - Under Construction"

@app.route('/analyzer')
def analyzer():
    return render_template('analyzer.html')

@app.route('/indexsy')
def indexsy():
    return render_template('indexsy.html')

if __name__ == '__main__':
    app.run(debug=True)
