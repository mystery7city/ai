from flask import Flask, render_template, request
from database.repository import get_emp_list, get_emp
import json
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/emp_list')
def emp_list():
    emp_list = get_emp_list()
    return json.dumps(emp_list)
@app.route('/emp/<int:empno>')
def emp(empno):
    emp = get_emp(empno)
    return json.dumps(emp)
if __name__ == '__main__':                  