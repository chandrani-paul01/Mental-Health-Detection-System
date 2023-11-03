from flask import Flask, render_template, request
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('mhd.html')

@app.route('/mhd', methods=['POST', 'GET'])
def show_iris():
    if request.method == 'POST':
        Age = float(request.form['age'])
        Gender = request.form['gen']
        self_employed = request.form['self_emp']
        family_history = request.form['fam_hist']
        work_interference = request.form['work_inter']
        no_employees = request.form['no_emp']
        remote_work = request.form['rem_work']
        tech_company= request.form['tech_comp']
        benefits = request.form['benefits']
        care_options =request.form['care_opt']
        well_program = request.form['well_prog']
        seek_help = request.form['seek_help']
        anonymity = request.form['anonimity']
        leave = request.form['leave']
        mental_health_consequence = request.form['mhc']
        phys_health_consequence = request.form['phc']
        coworkers = request.form['coworkers']
        supervisor = request.form['supervisors']
        mental_health_interview = request.form['mhi']
        phys_health_interview = request.form['phi']
        mental_vs_physical = request.form['m_vs_p']
        obs_consequence = request.form['obs_consq']

        df = pd.read_csv('survey.csv')
        df.drop(['comments'], axis= 1, inplace=True)
        df.drop(['state'], axis= 1, inplace=True)
        df.drop(['Country'], axis= 1, inplace=True)
        df.drop(['Timestamp'], axis= 1, inplace=True)
        X = df.drop('treatment',axis=1)
        
        transformer =  ColumnTransformer([('ordinal_encoder',OrdinalEncoder(),['Gender','self_employed',
                                  'family_history','work_interfere','no_employees','remote_work',
                                  'tech_company' ,'benefits','care_options','wellness_program',
                                  'seek_help','anonymity','leave','mental_health_consequence',
                                  'phys_health_consequence','coworkers','supervisor','mental_health_interview',
                                  'phys_health_interview','mental_vs_physical','obs_consequence'])],remainder='passthrough')
        
        transformer.fit(X)

        x2 = pd.DataFrame([[Age,Gender,self_employed,
                                  family_history,work_interference,no_employees,remote_work,
                                  tech_company ,benefits,care_options,well_program,
                                  seek_help,anonymity,leave,mental_health_consequence,
                                  phys_health_consequence,coworkers,supervisor,mental_health_interview,
                                  phys_health_interview,mental_vs_physical,obs_consequence]])
        
        



        import numpy as np
        x1 = transformer.transform(pd.DataFrame([[Age,Gender,self_employed,
                                  family_history,work_interference,no_employees,remote_work,
                                  tech_company ,benefits,care_options,well_program,
                                  seek_help,anonymity,leave,mental_health_consequence,
                                  phys_health_consequence,coworkers,supervisor,mental_health_interview,
                                  phys_health_interview,mental_vs_physical,obs_consequence]], columns=X.columns))

        
        rand_cls = pickle.load(open('rand_for.model', 'rb'))
        prediction = rand_cls.predict(x1)

        result_message = "You are predicted to have a mental health condition.Treatment is Encouraged." if prediction[0] == 1 else "You are predicted to not have a mental health condition."

        return render_template('mhd_res.html', result_message=result_message)

if __name__ == '__main__':
    app.run(debug=True)