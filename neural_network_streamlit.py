import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import base64
from tensorflow.keras.models import load_model


# Initialize session state for navigation
if "sb" not in st.session_state:
    st.session_state.sb = "Home"


# Sidebar navigation
sb = st.sidebar.radio('Main Menu', ['Home', 'Customer Churn Prediction', 'Learn More'], index=[
                      "Home", "Customer Churn Prediction", "Learn More"].index(st.session_state.sb))


# Home page
if sb == "Home":
    st.session_state.sb = "Home"

    st.title("📚 Welcome to the Customer Churn Predictor! 📊")

    st.image("E:/Guvi/VS_code/Final_Project/Main_folder/Neural_network-project/logo.jpg", width=700,
             caption="Your data-driven journey starts here!")

    st.subheader("🔍 Discover Insights")
    st.markdown(
        """
        - Explore **customer behaviors** using advanced Artificial Neural Networks (ANN) model.
        - Processed data from the **publishing industry** forms the backbone of this app.
        """
    )

    st.markdown(
        "🌟 *Predict customer churn effortlessly on the next page!* 😎")

    # Button to navigate to "Learn More"
    if st.button("Learn More About This App 🚀", help="Click to explore more about customer churn."):
        st.session_state.sb = "Learn More"

# Churn predictor
elif sb == 'Customer Churn Prediction':
    st.session_state.sb = "Customer Churn Prediction"

    st.title("Churn Predictor")

    left_column, right_column = st.columns(2)

    p1 = left_column.number_input(
        "**Enter the number of days the customer has been buying books:**", min_value=0, max_value=2000, value=0)
    p2 = right_column.number_input(
        "**Enter the number of days the customer has not returned to the shop:**", min_value=0, max_value=2000, value=0)

    selected_date = left_column.date_input("**Select an order date**")
    if selected_date:
        p3 = selected_date.day
        p4 = selected_date.month
        p5 = selected_date.year

    p6 = right_column.number_input(
        "**Enter the price of the book bought:**", min_value=0, max_value=1000, value=0)
    p7 = left_column.number_input(
        "**Enter the number of times the customer has ordered from your shop:**", min_value=0, max_value=500, value=0)

    # Load the model
    try:
        model = load_model(
            'E:/Guvi/VS_code/Final_Project/Main_folder/Neural_network-project/model.h5')
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        model = None

    # Prepare input data
    input_data = np.array(
        [[float(p2), float(p6), float(p1), float(p7), int(p3), int(p4), int(p5)]])

    if st.button('Predict'):
        pred = model.predict(input_data)
        predicted_class = np.argmax(pred, axis=1)[0]
        if predicted_class == 0:
            st.success("The customer has not churned 🎉😊🌟")
        else:
            st.warning(
                "The customer has churned. 🛑 Let's offer them a 10% discount 🎁 to win them back! 💡✨")

# Learn more page
elif sb == "Learn More":
    st.session_state.sb = "Learn More"
    st.title("Learn More About This App 🚀")
    st.markdown("""
        This app uses cutting-edge **Artificial Neural Networks (ANN)** to predict customer churn.
        By analyzing customer behavior data, the model helps you identify at-risk customers 
        and take proactive steps to retain them.
                """)

    left_col, right_col = st.columns(2)
    left_col.subheader("💡 Why Predict Churn?")
    left_col.markdown(
        """
        - **Save costs** by retaining valuable customers.
        - **Enhance loyalty** through targeted strategies.
        - Stay ahead with **data-driven decisions**.
        """)

    left_col.markdown("Go back to the [Home Page](#) to start exploring.")

    # Button to navigate back to "Home"
    if st.button("Back to Home"):
        st.session_state.sb = "Home"

    right_col.image(
        'E:/Guvi/VS_code/Final_Project/Main_folder/Neural_network-project/Customer_Churn.png', width=400)
