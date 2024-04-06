import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor  # Import XGBRegressor from xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import openai  # Import OpenAI library
from openai import OpenAI

# Set your OpenAI API key
api_key = "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"
client = OpenAI(api_key=api_key)

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("cleaned_listings.csv", encoding='latin1')  # Load dataset "Listings.csv"
    return data

# Get neighbourhoods for the selected city
def get_neighbourhoods(data, city):
    neighbourhoods = data[data['city'] == city]['neighbourhood'].unique()
    return neighbourhoods

# Train the model for price prediction
@st.cache_data
def train_model(data):
    # Remove rows with missing values
    data.dropna(inplace=True)

    # Select relevant features and target variable
    X = data[['city', 'neighbourhood', 'property_type', 'accommodates', 'bedrooms']]
    y = data['price']

    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X)
    feature_names = encoder.get_feature_names_out(input_features=X.columns)
    X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=feature_names)  # Convert to dense array

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded_df, y, test_size=0.2, random_state=42)

    # Train the model (using Gradient Boosting - XGBoost)
    model = XGBRegressor(random_state=42)  # Use XGBRegressor instead of RandomForestRegressor
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f'Mean Squared Error: {mse:.2f}')
    st.write(f'Mean Absolute Error: {mae:.2f}')

    return model, encoder

# Generate travel tips using OpenAI API
def generate_travel_tips(city, nationality, start_date, end_date):
    prompt = f"As a {nationality} traveler planning a trip to {city} from {start_date} to {end_date}, I need to know the visa requirements, travel insurance, packing essentials, and safety tips for my travel dates."
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        model="gpt-3.5-turbo",
        temperature=0,
    )
    print(response)
    return response.choices[0].message.content.strip()

# Main function to run the Streamlit app
def main():
    st.title("Accommodation Price Prediction and Travel Tips")

    # Load the data
    data = load_data()

    # Train the model
    model, encoder = train_model(data)

    # Sidebar inputs for user
    st.sidebar.header('User Input')
    city = st.sidebar.selectbox('City', data['city'].unique())
    nationality = st.sidebar.selectbox('Nationality', ['USA', 'UK', 'Canada', 'Australia', 'Germany'])  # Add more nationalities as needed
    start_date = st.sidebar.date_input('Start Date')
    end_date = st.sidebar.date_input('End Date')
    neighbourhoods = get_neighbourhoods(data, city)
    neighbourhood = st.sidebar.selectbox('Neighbourhood', neighbourhoods)
    property_type = st.sidebar.selectbox('Property Type', data['property_type'].unique())
    accommodates = st.sidebar.slider('Accommodates', min_value=1, max_value=20, value=4)
    bedrooms = st.sidebar.slider('Bedrooms', min_value=0, max_value=10, value=1)

    # Check if user has submitted the form
    if st.sidebar.button('Submit'):
        # One-hot encode user input
        input_data = pd.DataFrame({
            'city': [city],
            'neighbourhood': [neighbourhood],
            'property_type': [property_type],
            'accommodates': [accommodates],
            'bedrooms': [bedrooms]
        })
        input_encoded = encoder.transform(input_data)

        # Get feature names after one-hot encoding
        feature_names_encoded = encoder.get_feature_names_out(input_data.columns)

        # Create DataFrame with one-hot encoded features and correct column names
        input_encoded_df = pd.DataFrame(input_encoded.toarray(), columns=feature_names_encoded)  # Convert to dense array

        # Make prediction
        prediction = model.predict(input_encoded_df)

        # Display price prediction
        st.subheader('Price Prediction')
        st.write(f'The predicted price for the accommodation is: ${prediction[0]:.2f}')

        # Generate and display travel tips
        st.subheader('Travel Tips')
        travel_tips = generate_travel_tips(city, nationality, start_date, end_date)
        st.write(travel_tips)


if __name__ == "__main__":
    main()
