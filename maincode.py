#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 20:56:34 2020

@author: amir
"""

import streamlit as st
import pandas as pd
import numpy as np

import os
import joblib 

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import hashlib

from PIL import Image


from managedb import *

# Password 
def generate_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()


def verify_hashes(password,hashed_text):
	if generate_hashes(password) == hashed_text:
		return hashed_text
	return False

def load_images(image_name):
    img = Image.open(image_name)
    return st.image(img, width=800)

feature_names_best = [ 'fasting_bg', 'Gender', 'Polyuria', 'Polydipsia', 'Sudden_weight_loss', 'Weakness', 'Polyphagia', 'Genital_thrush', 'Visual_blurring','Itching', 'Irritability', 'Delayed_healing', 'Partial_paresis','Muscle_stiffness', 'Alopecia', 'Obesity']

gender_dict = {"Male":1,"Female":0}
feature_dict = {"Yes":1,"No":0}


def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 

def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return key

def get_fvalue(val):
	feature_dict = {"No":0,"Yes":1}
	for key,value in feature_dict.items():
		if val == key:
			return value 
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model
import lime
import lime.lime_tabular


def main():


    st.title('Pocket HealthRef')
    st.subheader('Welcome to ????')
    
    menu = ["Home","Login","SignUp"]
    submenu = [ "Visit Overview" , "Visit Summary", "Lab Results", "Medication", "Health Check" ]
    
    choice = st.sidebar.selectbox ("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        st.text( "some sort of information about our app")
      
    elif choice == "Login" :
        username = st.sidebar.text_input ("Username")
        password = st.sidebar.text_input ("password", type='password')
        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pswd = generate_hashes(password)
            result = login_user(username,verify_hashes(password,hashed_pswd))

            if result:
                    st.success("Welcome {}".format(username))
                    activity = st.selectbox("Activity", submenu)
                    if activity == ( "Plot"):    
                        st.subheader("Data Visualization Plot")
                        df = pd.read_csv("data/diabetesdata.csv")
                        st.dataframe(df)
                        if st.checkbox("Area Chart"):
                            all_columns = df.columns.to_list()
                            feat_choices = st.multiselect("Choose a Feature",all_columns)
                            new_df = df[feat_choices]
                            st.area_chart(new_df)

                        
		
                        
			
		
				
                    elif activity == "Health Check":
                            st.subheader("Instant feedback on your daily health measurments ")
                            fasting_bg = st.number_input("what was your fasting blood glucose?",7,580)   
                            pretty_result = {"Fasting blood glucose":fasting_bp,"Gender":Gender,"Polyuria":Polyuria,"Polydipsia":Polydipsia,"Sudden_weight_loss":Sudden_weight_loss,"Weakness":Weakness,"Polyphagia":Polyphagia,"Genital_thrush":Genital_thrush,"visual_blurring":Visual_blurring,"Itching":Itching,"Irritability":Irritability,"Delayed_healing":Delayed_healing,"Partial_paresis":Partial_paresis,"Muscle_stiffness":Muscle_stiffness,"Alopecia":Alopecia,"Obesity":Obesity}
                            st.json(pretty_result)
                            single_sample = np.array(feature_list).reshape(1,-1)
                            model_choice = st.selectbox("Select Model",["LR","KNN","DecisionTree"])
                            if st.button("Predict"):

                                    if model_choice == "KNN":
                                            loaded_model = load_model("models/knn_diabetes_model.pkl")
                                            prediction = loaded_model.predict(single_sample)
                                            pred_prob = loaded_model.predict_proba(single_sample)
                                    elif model_choice == "DecisionTree":
                                            loaded_model = load_model("models/decision_tree_clf_diabetes_model.pkl")
                                            prediction = loaded_model.predict(single_sample)
                                            pred_prob = loaded_model.predict_proba(single_sample)
                                    else:
                                            loaded_model = load_model("models/logistic_regression_diabetes_model.pkl")
                                            prediction = loaded_model.predict(single_sample)
                                            pred_prob = loaded_model.predict_proba(single_sample)
                                    
				
                                    if prediction == 0:
                                        st.warning("Patient is not Diabetic")
                                        pred_probability_score = {"Not Diabetic":pred_prob[0][0]*100,"Diabetic":pred_prob[0][1]*100}
                                        st.subheader("Prediction Probability Score using {}".format(model_choice))
                                        st.json(pred_probability_score)
                                        
                                    else:
                                        st.success("Patient is Diabetic")
                                        pred_probability_score = {"Not Diabetic":pred_prob[0][0]*100,"Diabetic":pred_prob[0][1]*100}
                                        st.subheader("Prediction Probability Score using {}".format(model_choice))
                                        st.json(pred_probability_score)                                  
                             
                            if st.checkbox("Interpret"):            
                                    if model_choice == "KNN":
                                        loaded_model = load_model("models/knn_diabetes_model.pkl")

			
                                    elif model_choice == "DecisionTree":
                                        loaded_model = load_model("models/decision_tree_clf_diabetes_model.pkl")
                            
                                    else:
                                        loaded_model = load_model("models/logistic_regression_diabetes_model.pkl")
					

			
                                        df1 = pd.read_csv("data/diabetesdata_binary.csv")
                                        x = df1[['Age','Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring','Itching', 'Irritability', 'delayed healing', 'partial paresis','muscle stiffness', 'Alopecia', 'Obesity']]
                                        feature_names = ['Age','Gender', 'Polyuria', 'Polydipsia', 'Sudden_weight_loss', 'Weakness', 'Polyphagia', 'Genital thrush', 'Visual_blurring','Itching', 'Irritability', 'Delayed_healing', 'Partial_paresis','Muscle_stiffness', 'Alopecia', 'Obesity']
                                        class_names = ['Negative','Positive']
                                        explainer = lime.lime_tabular.LimeTabularExplainer(x.values,feature_names=feature_names, class_names=class_names,discretize_continuous=True)
                                        exp = explainer.explain_instance(np.array(feature_list), loaded_model.predict_proba,num_features=16, top_labels=1)
                                        exp.show_in_notebook(show_table=True, show_all=False)
                                        # exp.save_to_file('lime_oi.html')
                                        st.write(exp.as_list())
                                        new_exp = exp.as_list()
                                        label_limits = [i[0] for i in new_exp]
                                                                                # st.write(label_limits)
                                        label_scores = [i[1] for i in new_exp]
                                        plt.barh(label_limits,label_scores)
                                        st.pyplot()
                                        plt.figure(figsize=(20,10))
                                        fig = exp.as_pyplot_figure()
                                        st.pyplot()	                                   					
					
					
					
					
					
        else:
    	        st.warning("Incorrect Username/Password, Please try again")

    elif choice == "SignUp" :
        new_username = st.text_input("User name")
        new_password = st.text_input("Password", type='password')

        confirm_password = st.text_input("Confirm Password", type='password')
        if new_password == confirm_password:
            st.success("Password Confirmed")
        else:
            st.warning("Passwords not the same")
            
        if st.button("Submit"):
             create_usertable()
             hashed_new_password = generate_hashes(new_password)
             add_userdata(new_username,hashed_new_password)
             st.success("You have successfully created a new account")
             st.info("Login to Get Started")
 
        








if __name__ == '__main__' :
    main()
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
