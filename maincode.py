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
    st.subheader('Welcome to Pocket HealthRef')
    
    menu = ["Home","Login","SignUp"]
    submenu = [ "Visit Overview" , "Visit Summary", "Lab Results", "Medication", "Health Check" ]
    
    choice = st.sidebar.selectbox ("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        st.text( "Pocket HealthRef is a web-based tool where you can access information about your recent :/ visits to the doctor’s office! Among other things, Patient HealthRef allows you to relisten to your visit with your doctor. If you’ve ever felt like you forgot some important information after a visit, Pocket HealthRef is just the tool for you! ")
      
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
                            pretty_result = {"Fasting blood glucose":fasting_bg}
                            st.json(pretty_result)
                            model_choice = st.selectbox("Select Model",["LR","KNN","Blood glucose Prection"])
                            if st.button("Predict"):

                                    if model_choice == "KNN":
                                            print("knn")
                                    elif model_choice == "Blood glucose Prection":
                                            if fasting_bg < 140: 
                                                print("normal") 
						
						
						
                                
                                           
                                    
				                                  
                             
                            if st.checkbox("Interpret"):            
                                    if model_choice == "KNN":
                                        loaded_model = load_model("models/knn_diabetes_model.pkl")

			
                                    elif model_choice == "DecisionTree":
                                        loaded_model = load_model("models/decision_tree_clf_diabetes_model.pkl")
                            
                                    else:
                                        loaded_model = load_model("models/logistic_regression_diabetes_model.pkl")
					

			
                                                                       					
					
					
					
					
					
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
