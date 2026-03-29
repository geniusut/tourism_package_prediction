
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# =========================================
# Load Model from Hugging Face
# =========================================
model_repo_id = "geniusut/tourism-package-prediction-project"

model_path = hf_hub_download(
    repo_id=model_repo_id,
    filename="model.pkl"
)

model = joblib.load(model_path)

# =========================================
# Streamlit UI
# =========================================
st.title("Wellness Tourism Package Prediction")

st.write("""
Predict whether a customer will purchase the wellness tourism package.
Fill in the details below:
""")

# =========================================
# Mapping dictionaries
# =========================================
TypeofContact_map = {"Company Invited": 0, "Self Enquiry": 1}
Occupation_map = {"Salaried": 0, "Free Lancer": 1, "Small Business": 2}
Gender_map = {"Male": 0, "Female": 1}
ProductPitched_map = {"Basic": 0, "Standard": 1, "Deluxe": 2}
MaritalStatus_map = {"Single": 0, "Married": 1, "Divorced": 2}
Designation_map = {"Manager": 0, "Executive": 1, "VP": 2}

# =========================================
# Inputs
# =========================================
Age = st.number_input("Age", 18, 70, 30)

TypeofContact = st.selectbox("Type of Contact", list(TypeofContact_map.keys()))
TypeofContact = TypeofContact_map[TypeofContact]

CityTier = st.selectbox("City Tier", [1, 2, 3])

DurationOfPitch = st.number_input("Duration Of Pitch", 0.0, 60.0, 10.0)

Occupation = st.selectbox("Occupation", list(Occupation_map.keys()))
Occupation = Occupation_map[Occupation]

Gender = st.selectbox("Gender", list(Gender_map.keys()))
Gender = Gender_map[Gender]

NumberOfPersonVisiting = st.number_input("Number Of Persons Visiting", 1, 10, 2)

NumberOfFollowups = st.number_input("Number Of Followups", 0, 10, 2)

ProductPitched = st.selectbox("Product Pitched", list(ProductPitched_map.keys()))
ProductPitched = ProductPitched_map[ProductPitched]

PreferredPropertyStar = st.number_input("Preferred Property Star", 1, 5, 3)

MaritalStatus = st.selectbox("Marital Status", list(MaritalStatus_map.keys()))
MaritalStatus = MaritalStatus_map[MaritalStatus]

NumberOfTrips = st.number_input("Number Of Trips", 0, 10, 2)

Passport = st.selectbox("Has Passport", [0, 1])

PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", 1, 5, 3)

OwnCar = st.selectbox("Own Car", [0, 1])

NumberOfChildrenVisiting = st.number_input("Children Visiting", 0, 5, 0)

Designation = st.selectbox("Designation", list(Designation_map.keys()))
Designation = Designation_map[Designation]

MonthlyIncome = st.number_input("Monthly Income", 10000, 200000, 50000)

# =========================================
# Convert to DataFrame
# =========================================
input_df = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

# =========================================
# Prediction
# =========================================
if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("Customer is likely to purchase the package ✅")
    else:
        st.error("Customer is unlikely to purchase ❌")
