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

import time
import datetime
from datetime import datetime, date, time



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
    return st.image(img, width=600)
def load_images2(image_name):
    img = Image.open(image_name)
    return st.image(img, width=400)

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
    submenu = [ "Visit Overview" , "Visit Summary", "Medications", "Health Check" ]
    
    choice = st.sidebar.selectbox ("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        st.markdown( "Pocket HealthRef is a web-based tool where you can access information about your recent visits to the doctor???s office!\n Among other things, Pocket HealthRe allows you to relisten to your visit with your doctor. If you???ve ever felt like you forgot some important information after a visit, Pocket HealthRef is just the tool for you! ")
        Image.open('logo1.png').convert('RGB').save('logo2.png')
        c_image7 = 'logo2.png'
        load_images2(c_image7)
        st.markdown("**To start using Pocket HealthRef, please follow these steps:**")
        st.markdown(" New Users: Create an account by clicking the SignUp option from the dropdown menu on the left side of your screen") 
        st.markdown(" Returning Users: Login to your account by clicking on the Login option from the dropdown menu.")
        st.markdown(" At this point, you should be able to see the data that your doctor has added to your account.")
        


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
                        st.markdown("** My Doctor???s Contact Information: **")
                        st.markdown(" Address: 4444 West Coast Dr., Richmond, VA 23220")
                        st.markdown(" Phone Number: (804) 987-6554")
                        st.markdown(" Office Hours: 8 am to 3 pm, Monday to Thursday")
                        st.markdown(" If you have any question, or if you need to reschedule your next appointment, please call me using the phone number above or use the following calender to request a new appoinment.")
                        st.header("Request to book a new appoinment here:")
                        st.date_input('Office Availability')
                        if st.button("Submit"):
                            st.success("We have received your appoinment request. We will  be in touch within 24 hours of receiving your request to confirm your appoinment.")				
                    elif activity == ( "Visit Summary"): 
                        st.subheader("Visit Summary")
                        st.markdown("Date: 10/10/2021")
                        st.markdown(" **Doctor???s Notes:**")
                        st.markdown(" Patient presented for a routine checkup. Vitals are all within range considering diabetes mellitus. Continue medications, exercise routine, and diet. Weight loss is at a steady and healthy pace. There were no observations of side effects from medications.")
                        st.markdown(" **Recent Vitals:**")
                        st.markdown(" Blood Pressure: 130/80 mmHg")
                        st.markdown(" Blood Glucose: 90 mg/dL")
                        st.markdown(" Heart Rate: 90 beats per minute")
                        st.markdown(" Weight: 243 lbs")
                        st.write("Please find your full visit summary below:")

                        Image.open('SampleVisitSummary.png').convert('RGB').save('SampleVisitSummary2.png')
                        c_image = 'SampleVisitSummary2.png'
                        load_images(c_image)
                        st.markdown(" **Lab Results**")
                        st.markdown(" Lab results are all normal. ")
                        st.write("Please find your full lab results below:")

                        Image.open('labresults.png').convert('RGB').save('labresults2.png')
                        c_image2 = 'labresults2.png'
                        load_images(c_image2)

                    elif activity == "Medications":
                        st.subheader("Medications")
                        st.markdown("**Insulin Glargine [Lantus SoloStar] 100 mg/mL for Type 2 Diabetes**")
                        st.markdown("**Direction:** Inject ten (10) units into the side of your lower abdomen once a day. Injection instructions are included with the medication pen.")
                        st.markdown("**Storage:** Refrigerate medication pens until ready to inject. Take one pen out of the fridge and let sit for 5 minutes before injection. Do not store medication pen outside of the refrigerator.")
                        Image.open('Lantus3.png').convert('RGB').save('lantus2.png')
                        c_image4 = 'lantus2.png'
                        load_images2(c_image4)
                        st.markdown("**Amlodipine 5mg for High Blood Pressure**")
                        st.markdown("**Direction:** Take one (1) tablet by mouth once a day for hypertension.")
                        st.markdown("**Storage:** Store at room temperature away from moisture or direct sunlight.")                      
                        Image.open('amlodipine.png').convert('RGB').save('amlodipine2.png')
                        c_image3 = 'amlodipine2.png'
                        load_images2(c_image3)	
                        user_input = st.text_input("Enter your phone number to receive daily reminders when your medication is due to take.")
                        if st.button("Submit"):
                            st.success("You will receive text message reminders whenever your medication is due to take. text STOP to opt out.")
                        st.subheader("**My Pharmacy Contacts:**")
                        st.markdown("**Pharmacist???s Name:** Dr. Uwu Lepolski")
                        st.markdown("**Address:** 4448 West Coast Dr., Richmond, VA 23220")
                        st.markdown("**Phone Number:** (804) 987-6555")
                        st.markdown("Call this number if you have questions about your current medications.")

			
			
			
			
			
			
                    elif activity == "Health Check":
                            int_val3 = st.number_input('What was your fasting blood glucose value today?', min_value=0, max_value=250, value=0, step=1)
                            if int_val3 > 130 :
                            	st.info("Your fasting blood glucose value is above the recommended goal.")      
                            if int_val3 < 70 and int_val3 > 0:
                            	st.info("Your fasting blood glucose value is below the recommended goal. recheck your blood glucose in 15 minutes, if the value is still low take an equivalent of 15 g of sugar (e.g. half a can of soda)")
                            if int_val3 > 70 and int_val3 < 130 :
                            	st.info("Your fasting blood glucose value is within the recommended goal.")
                            st.markdown("**Your fasting blood glucose trend:**")
                            st.markdown("The recommended fasting blood glucose range is within **70???130** mg/dl for you.")
                            df = pd.read_csv("Glucose.csv")
                            st.line_chart(df)
                            int_val = st.number_input('What was your systolic blood pressure today?', min_value=0, max_value=250, value=0, step=1)
                            int_val2 = st.number_input('What was your diastolic blood pressure today?', min_value=0, max_value=250, value=0, step=1)
                            if int_val > 130 or int_val2 > 80:
                            	st.info("Your blood pressure is above the recommended goal.")      
                            if int_val < 130 and int_val > 5 and int_val2 < 80 and int_val2 >5:
                            	st.info("Your blood pressure is within the recommended goal.")
			
                            st.markdown("**Your systolic and diastolic blood pressure trend:**")
                            st.markdown("The recommended systolic and diastolic blood pressure values for you are less than **130** mmHg and **80** mmHg respectively.")
                            df2 = pd.read_csv("bp.csv")
                            st.bar_chart(df2)
				
			
						
						
						
                                
                                           
                                    
				                                  
					
					
					
					
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
