# Import libraries
import pandas as pd      
import numpy as np       
import pickle
import streamlit as st 
from catboost import CatBoostRegressor
from sklearn.feature_extraction import DictVectorizer    
import gdown 
import os

# Columns
columns = ['State', 'City', 'Property_Type', 'BHK', 'Size_in_SqFt',
        'Price_per_SqFt', 'Year_Built', 'Furnished_Status',
       'Floor_No', 'Total_Floors', 'Age_of_Property', 'Nearby_Schools',
       'Nearby_Hospitals', 'Public_Transport_Accessibility', 'Parking_Space',
       'Security', 'Amenities', 'Facing', 'Owner_Type', 'Availability_Status']

# Load the Random Forest model
url = "https://drive.google.com/uc?id=1lS1CzNT0v97gT97KomMtoi1CecUwdpU-"
file_id = "1lS1CzNT0v97gT97KomMtoi1CecUwdpU-"
output = 'random_forest_model_new.pkl'
gdown.download(url, output, quiet=False)
with open(output, 'rb') as file:
    model_rf = pickle.load(file)

# Load the Gradient Boosting model
pickle_gb = open('gradient_boosting_model_new.pkl', 'rb')
model_gb = pickle.load(pickle_gb)

# Load the CatBoost model
pickle_cb = open('cat_boost_model_new.pkl', 'rb')
model_cb = pickle.load(pickle_cb)

# Load the Dictionary Vectorizer 
pickle_cb = open('dict_vectorizer.pkl', 'rb')
dv = pickle.load(pickle_cb)

# Function to perform prediction
def predict(data_frame):
    predict_rf = model_rf.predict(data_frame)
    predict_gb = model_gb.predict(data_frame)
    stack_pred = np.column_stack((predict_rf, predict_gb))
    predict_cb = model_cb.predict(stack_pred)
    return predict_cb


# Function to accept inputs and predict from inputs
def main():
    # Set up the application
    st.title('House Price Predictor')
    html_temp = '''
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> House Price Prediction App </h2>
    </div>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)

    # Specify the inputs
    st.write("Please enter values for all the parameters!!!")
    state = st.text_input("State: ") 
    city = st.text_input("City: ") 
    property_type = st.radio("Property type: ",
                                  ["Apartment", "Villa", "Independent"]).lower()
    bhk = st.text_input("BHK: ")
    size_in_sqft = st.text_input("Size in sqft: ")
    price_per_sqft = st.text_input("Price per sqft: ")
    year_built = st.text_input("Year built: ") 
    furnished_status = st.radio("Furnished status: ",
                                ["Furnished", "Semi-furnished", "Unfurnished"]).lower()
    floor_no = st.text_input("Floor number: ")
    total_floors = st.text_input("Total number of floors: ")
    age_of_property = st.text_input("Age of property: ")
    nearby_schools = st.text_input("Number of nearby schools: ")
    nearby_hospitals = st.text_input("Number of nearby hospitals: ")
    public_transport_accessibility = st.radio("Public transport accessibility: ",
                                              ["Low", "Medium", "High"]).lower()
    parking_space = st.radio("Parking space available?: ",
                                  ["Yes", "No"]).lower()
    security = st.radio("Security available?: ",
                             ["Yes", "No"]).lower()
    selected_amenities = st.multiselect(
    "Select amenities: ",
    ["Clubhouse", "Gym", "Pool", "Garden", "Playground"]
    )
    amenities_list = [amenity.lower() for amenity in selected_amenities]
    amenities_str = ", ".join(amenities_list)
    facing = st.radio("Facing: ",
                      ["North", "South", "East", "West"]).lower()
    owner_type = st.radio("Owner type: ", ["Owner", "Builder", "Broker"]).lower()
    status = st.radio("Status: ", ["Under_Construction", "Ready_to_Move"]).lower()


    # Combine into an input data
    data_dict = {
        'State': [state],
        'City': [city],
        'Property_Type': [property_type],
        'BHK': [bhk],
        'Size_in_SqFt': [size_in_sqft], 
        'Price_per_SqFt': [price_per_sqft],
        'Year_Built': [year_built],
        'Furnished_Status': [furnished_status],
        'Floor_No': [floor_no],
        'Total_Floors': [total_floors],
        'Age_of_Property': [age_of_property],
        'Nearby_Schools': [nearby_schools],
        'Nearby_Hospitals': [nearby_hospitals],
        'Public_Transport_Accessibility': [public_transport_accessibility],
        'Parking_Space': [parking_space], 
        'Security': [security],
        'Amenities': [amenities_str],
        'Facing': [facing],
        'Owner_Type': [owner_type],
        'Availability_Status': [status],  
    }
    df = pd.DataFrame(data_dict)
    df_dict = df.to_dict(orient='records')
    df_dict = dv.transform(df_dict)

    # Predict button
    if st.button('Predict'):
        result = predict(df_dict)
        st.success(f"House price is {result[0]}.")

# Run the application
if __name__ == '__main__': 
    main() 




