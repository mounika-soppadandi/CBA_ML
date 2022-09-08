#!/usr/bin/env python
# coding: utf-8

# In[20]:


# importing necessary libraries
import joblib
import pandas as pd
import streamlit as st



#load the model
KMeansCls = joblib.load('clust1_cba.pkl')

#page configuration
st.set_page_config(page_title = 'Customer Behaviour Analysis', layout='centered')
st.title('Customer Behaviour Analysis')

# customer segmentation function
def segment_customers(input_data):
    
    prediction=KMeansCls.predict(pd.DataFrame(input_data, columns=['Income', 'Age', 'Month_Customer', 'TotalSpendings', 'Children']))
    print(prediction)
    pred_1 = 0
    if prediction == 0:
            pred_1 = 'Highly Active Customer'

    elif prediction == 1:
            pred_1 = 'Moderately Active Customer'

    elif prediction == 2:
            pred_1 = 'Least Active Customer'

    return pred_1
def main():
    st.image("""https://d3caycb064h6u1.cloudfront.net/wp-content/uploads/2021/08/customerbehavior-scaled.jpg""")
    
    Income = st.text_input("Enter Household Income 5 digitnumber")
    Children = st.radio ( "Select Number Of Kids In Household", ('0', '1','2','3') )
    Month_Customer = st.text_input( "Enter the number of months customer affiliated with the company ")
    Age = st.slider ( "Select Age", 18, 85 )
    TotalSpendings= st.text_input( "Enter TotalSpendings")
    
    
    result = ""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Analyze Customer"):
        result=segment_customers([[Income,Age,Month_Customer,TotalSpendings,Children]])
    
    st.success(result)
if __name__ == '__main__':
        main ()






