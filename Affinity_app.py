import streamlit as st
import pickle as pk
import pandas as pd
from category_encoders import OrdinalEncoder

st.title("Affinity Card Application")

def transform_input(record):
    """
    This takes in all the data and transform it into a dataframe.
    """
    # convert the record to a dataframe
    df = pd.DataFrame(record, index=[0])

    # Create a dictionary.
    gender = {"Female": 0, "Male": 1}
    # Replace all the values in the "Customer Gender" by the gender dictionary.
    df["CUST_GENDER"] = df["CUST_GENDER"].replace(gender)
    # Get the corresponding number for the cn variable.
    cn = ['United States of America', 'Brazil', 'Argentina', 'Germany',
        'Italy', 'New Zealand', 'Australia', 'Poland', 'Saudi Arabia',
        'Denmark', 'Japan', 'China', 'Canada', 'United Kingdom',
        'Singapore', 'South Africa', 'France', 'Turkey', 'Spain']
    num_cn = []
    for i in range(0, len(cn)):
        num_cn.append(i)
    cn_dict = dict(zip(cn, num_cn))  # Create a dictionary having the country name as key and the value as number.
    df["COUNTRY_NAME"]= df["COUNTRY_NAME"].replace(cn_dict)

    # cust_income.
    df["CUST_INCOME_LEVEL"] = df["CUST_INCOME_LEVEL"].str.split(":", expand=True)[0] 
    df["CUST_INCOME_LEVEL"] = df["CUST_INCOME_LEVEL"].apply(lambda c: 1 if c in ["A", "B", "C", "D"] else(2 if c in ["E", "F", "G", "H"] else 3))

    edu_level = {"Presch.": 1, '1st-4th': 1, '5th-6th': 1, '7th-8th': 2, '9th': 3, '10th': 3, '11th':3, '12th':3,
            'Masters': 4, 'Bach.':4, 'HS-grad':4, '< Bach.':4, 'Profsc':4, 'Assoc-A':4, 'Assoc-V':4, "PhD":4}
    df["EDUCATION"] = df["EDUCATION"].replace(edu_level)

    df["HOUSEHOLD_SIZE"] = df["HOUSEHOLD_SIZE"].astype("int")

    # Converting the remaining object datatype to numbers.
    ordinal = OrdinalEncoder()    # Initialize the OrdinalEncoder class.
    df[["CUST_MARITAL_STATUS", "OCCUPATION"]] = ordinal.fit_transform(df[["CUST_MARITAL_STATUS", "OCCUPATION"]])   # Fit and transform the object datatype.
    return df


# Setting the input for all the values.
with st.form("my_form", clear_on_submit=True):
    gender = st.radio("GENDER", ("Male", "Female"), horizontal=True)
    age = st.number_input("AGE", step=1)
    income = st.selectbox("INCOME LEVEL", ('J: 190,000 - 249,999', 'I: 170,000 - 189,999',
       'H: 150,000 - 169,999', 'B: 30,000 - 49,999',
       'K: 250,000 - 299,999', 'L: 300,000 and above',
       'G: 130,000 - 149,999', 'C: 50,000 - 69,999',
       'E: 90,000 - 109,999', 'D: 70,000 - 89,999',
       'F: 110,000 - 129,999', 'A: Below 30,000'))
    col1, col2 = st.columns(2)
    col3, col4, col5= st.columns(3)
    with col1:
        marital_status = st.selectbox(
                                    "MARITAL STATUS", 
                                    ('NeverM', 'Married', 'Divorc.', 'Mabsent', 'Separ.', 'Widowed', 'Mar-AF'))
        education = st.selectbox("EDUCATION", 
                            ("Presch.", '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th',
                            'Masters', 'Bach.', 'HS-grad', '< Bach.', 'Profsc', 'Assoc-A', 'Assoc-V', "PhD"))
        house_size = st.number_input("HOUSE HOLD SIZE", step = 1)
    with col2:
        country = st.selectbox("COUNTRY",
                            ('United States of America', 'Brazil', 'Argentina', 'Germany',
                            'Italy', 'New Zealand', 'Australia', 'Poland', 'Saudi Arabia',
                            'Denmark', 'Japan', 'China', 'Canada', 'United Kingdom',
                            'Singapore', 'South Africa', 'France', 'Turkey', 'Spain'))
        occupation = st.selectbox("OCCUPATION",
                            ('Prof.', 'Sales', 'Cleric.', 'Exec.', 'Other', 'Farming',
                            'Transp.', 'Machine', 'Crafts', 'Handler', 'Unemployed', 'Protec.',
                            'TechSup', 'House-s', 'Armed-F'))
        resident_year = st.number_input('YRS_RESIDENCE', step= 1)
    with col3:
        fpm = st.radio('FLAT_PANEL_MONITOR', (0, 1), horizontal=True)
        ybg = st.radio('Y_BOX_GAMES', (0, 1), horizontal=True)
    with col5:
        bpd = st.radio('BULK_PACK_DISKETTES', (0, 1), horizontal=True)    
        ba = st.radio('BOOKKEEPING_APPLICATION', (0, 1), horizontal=True) 
    with col4:
        htp = st.radio('HOME_THEATER_PACKAGE', (0, 1), horizontal=True)
        odsk = st.radio('OS_DOC_SET_KANJI', (0, 1), horizontal=True)
        if st.form_submit_button("PREDICT", type="primary"):
            data_dict = {
                        'CUST_GENDER': gender, 'AGE':age, 'CUST_MARITAL_STATUS':marital_status, 'COUNTRY_NAME':country,
                        'CUST_INCOME_LEVEL':income, 'EDUCATION': education, 'OCCUPATION': occupation, 'HOUSEHOLD_SIZE': house_size,
                        'YRS_RESIDENCE':resident_year, 'BULK_PACK_DISKETTES': bpd,
                        'FLAT_PANEL_MONITOR':fpm, 'HOME_THEATER_PACKAGE':htp, 'BOOKKEEPING_APPLICATION': ba,
                        'Y_BOX_GAMES': ybg, 'OS_DOC_SET_KANJI': odsk
                        }
            df = transform_input(data_dict)
            model = pk.load(open("LogisticRegression.pkl", "rb"))
            prediction = model.predict(df)
            st.success(f"Prediction of the Affinity Card is {prediction[0]}")