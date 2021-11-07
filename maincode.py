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
import base64

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


    st.title('                 Pocket HealthRef')
    st.subheader('Welcome to Pocket HealthRef')
    
    menu = ["Home","Login","SignUp"]
    submenu = [ "Visit Overview" , "Visit Summary", "Lab Results", "Medication", "Health Check" ]
    
    choice = st.sidebar.selectbox ("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        st.markdown( "Pocket HealthRef is a web-based tool where you can access information about your recent visits to the doctor’s office!\n Among other things, Patient HealthRef allows you to relisten to your visit with your doctor. If you’ve ever felt like you forgot some important information after a visit, Pocket HealthRef is just the tool for you! ")
        st.markdown("**To start using Pocket HealthRef, please follow these steps:**")
        st.markdown(" New Users: Create an account by clicking the SignUp option from the dropdown menu on the left side of your screen") 
        st.markdown(" Login to your account by clicking on the Login option from the dropdown menu.")
        st.markdown(" At this point, you should be able to see the data that your doctor has added to your account.")
        st.markdown(" The next time you go to your doctor’s office, make sure to sign in to Pocket HealthRef on their computer so that your doctor can update your information.")


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
                    if activity == ( "Visit Overview"): 
                        st.subheader("Visit Overview")
                        st.markdown("** What are the changes after my previous visit? **")
                        st.markdown(" No change in medication.")
                        st.markdown(" No change in diet.")
                        st.markdown(" No change in exercise recommendations.")
                        st.markdown("** My next appointment: **")
                        st.markdown(" General Checkup on 12/10/2022 at 10:00 am. Make sure not to eat anything that morning. You can drink water.")	
                        st.markdown("** My Doctor’s Contact Information: **")
                        st.markdown(" Address: 4444 West Coast Dr., Richmond, VA 23220")
                        st.markdown(" Phone Number: (804) 987-6554")
                        st.markdown(" Office Hours: 8 am to 3 pm, Monday to Thursday")
                        st.markdown(" If you have any questions, or if you need to reschedule your next appointment, please call me using the phone number above!")
                    elif activity == ( "Visit Summary"): 
                        st.subheader("Visit Summary")
                        st.markdown("Date: 10/10/2021")
                        st.markdown(" **Doctor’s Notes:**")
                        st.markdown(" Patient presented for a routine checkup. Vitals are all within range considering diabetes mellitus. Continue medications, exercise routine, and diet. Weight loss is at a steady and healthy pace. There were no observations of side effects from medications.")
                        st.markdown(" **Recent Vitals:**")
                        st.markdown(" Blood Pressure: 130/80 mmHg")
                        st.markdown(" Blood Glucose: 90 mg/dL")
                        st.markdown(" Heart Rate: 90 beats per minute")
                        st.markdown(" Weight: 243 lbs")
                        st.write("check out this [link](https://github.com/amirmbhd/VCUHEALTHHACK/blob/b919d2a4dd77a8a2d9c9d9dcf283a49b07ea32c7/Sample%20Visit%20Summary.pdf)")
                        st.markdown(" **Lab Results**")
                        st.markdown(" Lab results are all normal. ")
                        def show_pdf(file_path):
			with open(SampleVisitSummary.pdf,"rb") as f:
				base64_pdf = base64.b64encode(f.read()).decode('utf-8')
			pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
			st.markdown(pdf_display, unsafe_allow_html=True)


show_pdf("my_pdf.pdf")


			
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
