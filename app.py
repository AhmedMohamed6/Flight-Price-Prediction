# Import necessary libraries.
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature names
model = joblib.load("model.pkl")
inputs = joblib.load("inputs.pkl")

# Define a function for making predictions
def predict_flight_price(Airline, Source, Destination, Total_Stops, Additional_Info, month, day, 
                          Dep_Period, Arrival_Period, Duration_minutes):
    # Create a DataFrame with columns based on feature names.
    df = pd.DataFrame(columns=inputs)
    # Set the provided input values in the DataFrame.
    df.at[0, 'Airline'] = Airline
    df.at[0, 'Source'] = Source
    df.at[0, 'Destination'] = Destination
    df.at[0, 'Total_Stops'] = Total_Stops
    df.at[0, 'Additional_Info'] = Additional_Info
    df.at[0, 'month_of_Journey'] = month
    df.at[0, 'day_of_Journey'] = day
    df.at[0, 'Dep_Period'] = Dep_Period
    df.at[0, 'Arrival_Period'] = Arrival_Period
    df.at[0, 'Duration_minutes'] = Duration_minutes
    # Use the model to make predictions.
    price_prediction = model.predict(df)
    return price_prediction[0]

# Define the main function for the Streamlit app.
def main():
    # Create a Streamlit web interface.
    st.title("Flight Price Prediction")

    # Create input fields for user input
    features = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Additional_Info', 'month', 'day',
                'Dep_Period', 'Arrival_Period', 'Duration_minutes']

    input_values = [st.text_input(feature) for feature in features]

    if st.button("Predict"):
        # Call the prediction function and display the result
        price_result = predict_flight_price(*input_values)
        st.write(f"Predicted Flight Price: {price_result} INR")

if __name__ == "__main__":
    main()
