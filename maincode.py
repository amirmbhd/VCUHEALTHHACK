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

feature_names_best = [ 'Age', 'Gender', 'Polyuria', 'Polydipsia', 'Sudden_weight_loss', 'Weakness', 'Polyphagia', 'Genital_thrush', 'Visual_blurring','Itching', 'Irritability', 'Delayed_healing', 'Partial_paresis','Muscle_stiffness', 'Alopecia', 'Obesity']

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


    st.title('Diabetes Analysis App')
    st.subheader('Welcome to The Future')
    
    menu = ["Home","Login","SignUp"]
    submenu = [ "Plot" , "Prediction", "Analytics" ]
    
    choice = st.sidebar.selectbox ("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        st.text( "Are you on path to Diabetes? Let\'s findout")
        c_image = 'diab.png'
        load_images(c_image)
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

                        
		
                        
			
		
				
                    elif activity == "Prediction":
                            st.subheader("Predictive Analytics")
                            Age = st.number_input("Age",7,80)                        	
                            Gender = st.radio("Gender",tuple(gender_dict.keys()))
                            Polyuria = st.radio("Do you have Polyuria?",tuple(feature_dict.keys()))
                            Polydipsia = st.radio("Do you have Polydipsia?",tuple(feature_dict.keys()))
                            Sudden_weight_loss = st.radio("Hve you recently had sudden weight loss?",tuple(feature_dict.keys()))
                            Weakness = st.radio("Do you usually have muscle weakness?",tuple(feature_dict.keys()))
                            Polyphagia = st.radio("Have you recently experienced excessive hunger or increased appetite?",tuple(feature_dict.keys()))
                            Genital_thrush = st.radio("Do you have Genital thrush?",tuple(feature_dict.keys()))
                            Visual_blurring = st.radio("Have you recently experienced visual blurring?",tuple(feature_dict.keys()))
                            Itching = st.radio("Have you recently experienced Unexplained Itching?",tuple(feature_dict.keys()))
                            Irritability = st.radio("Have you recently experienced excessive Irritability?",tuple(feature_dict.keys()))
                            Delayed_healing = st.radio("Have you recently experienced delayed wound healing?",tuple(feature_dict.keys()))
                            Partial_paresis = st.radio("Have you recently experienced partial paresis?",tuple(feature_dict.keys()))
                            Muscle_stiffness = st.radio("Have you recently experienced muscle stiffness?",tuple(feature_dict.keys()))
                            Alopecia = st.radio("Do you have Alopecia?( patchy hair loss)",tuple(feature_dict.keys()))
                            Obesity = st.radio("Do you have Obesity based on your BMI?",tuple(feature_dict.keys()))                      
                            feature_list = [Age,get_value(Gender,gender_dict),get_fvalue(Polyuria),get_fvalue(Polydipsia),get_fvalue(Sudden_weight_loss),get_fvalue(Weakness),get_fvalue(Polyphagia),get_fvalue(Genital_thrush),get_fvalue(Visual_blurring),get_fvalue(Itching),get_fvalue(Irritability),get_fvalue(Delayed_healing),get_fvalue(Partial_paresis),get_fvalue(Muscle_stiffness),get_fvalue(Alopecia),get_fvalue(Obesity)]
                            st.write(len(feature_list))			
                            pretty_result = {"Age":Age,"Gender":Gender,"Polyuria":Polyuria,"Polydipsia":Polydipsia,"Sudden_weight_loss":Sudden_weight_loss,"Weakness":Weakness,"Polyphagia":Polyphagia,"Genital_thrush":Genital_thrush,"visual_blurring":Visual_blurring,"Itching":Itching,"Irritability":Irritability,"Delayed_healing":Delayed_healing,"Partial_paresis":Partial_paresis,"Muscle_stiffness":Muscle_stiffness,"Alopecia":Alopecia,"Obesity":Obesity}
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
