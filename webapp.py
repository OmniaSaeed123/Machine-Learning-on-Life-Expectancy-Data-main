
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import base64
import pickle

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
#from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
warnings.filterwarnings("ignore")
st.write("""
# Get Your Life Expectancy
### Let's Go!
""")
Data=pd.read_csv("Life_Expectancy_Data.csv")
def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
#add_bg_from_local("back.jpg")        
im=Image.open("average-us-life-expectancy-2.webp")
st.sidebar.image(im)
st.sidebar.header("Input data for Prediction")
st.sidebar.write("Kindly answer the following questions to check your life expectancy :")
def InputData():
    
    selectC=st.sidebar.selectbox("Select Country:",["Afghanistan","Haiti","Belgium","Germany","Russian Federation","Romania","Other"])
    if(selectC=="Afghanistan"): Country=0
    elif(selectC=="Haiti"): Country=1
    elif(selectC=="Belgium"): Country=2
    elif(selectC=="Germany"): Country=3
    elif(selectC=="Other"): Country=4
    elif(selectC=="Russian Federation"): Country=5
    elif(selectC=="Romania"): Country=6
    else: Country =7

    selectD=st.sidebar.selectbox("Select Status:",["Developing","Developed"])
    if(selectD=="Developing"): Status=0
    elif(selectC=="Developed"): Status=1
    else: Status =3


    selectYear=selectL=st.sidebar.selectbox("Select Year:",["2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015"])
    if(selectYear=="2000"): Year=0
    elif(selectYear=="2001"): Year=1
    elif(selectYear=="2002"): Year=2
    elif (selectYear == "2003"): Year=3
    elif (selectYear == "2004"): Year=4
    elif (selectYear == "2005"): Year=5
    elif (selectYear == "2006"): Year=6
    elif (selectYear == "2007"): Year = 7
    elif (selectYear == "2008"): Year = 8
    elif (selectYear == "2009"): Year = 9
    elif (selectYear == "2010"): Year = 10
    elif (selectYear == "2011"): Year = 11
    elif (selectYear == "2012"): Year = 12
    elif (selectYear == "2013"): Year = 13
    elif (selectYear == "2014"): Year = 14
    elif (selectYear == "2015"): Year = 15
    else: Year=17
    Adult_Mortality=st.sidebar.number_input("Adult Mortality:")
    infant_deaths=st.sidebar.number_input("infant_deaths:")
    percentage_expenditure=st.sidebar.number_input("percentage expenditure:")
    under_five_deaths=st.sidebar.number_input("under-five deaths:")
    Total_expenditure=st.sidebar.number_input("Total expenditure:")
    GDP=st.sidebar.number_input("GDP:")
    Income_composition_of_resources=st.sidebar.number_input("Income composition of resources:")
    Hepatitis_B=st.sidebar.number_input("Hepatitis B:")
    Measles=st.sidebar.number_input("Measles:")
    BMI=st.sidebar.number_input(" BMI:")
    Polio=st.sidebar.number_input("Polio")
    HIV_AIDS=st.sidebar.number_input("HIV/AIDS")
    Alcohol=st.sidebar.number_input("Alcohol")
    thinness_1_19_years=st.sidebar.number_input(" thinness  1-19 years")
    thinness_5_9_years=st.sidebar.number_input(" thinness 5-9 years")
    Schooling=st.sidebar.number_input("Schooling")

    data={"Adult_Mortality":Adult_Mortality,
         "infant_deaths":infant_deaths,
         "percentage_expenditure":percentage_expenditure,
         
         "under_five_deaths":under_five_deaths,
         "Total_expenditure":Total_expenditure,
         "GDP":GDP,
         "Income_composition_of_resources":Income_composition_of_resources,
        
         "Hepatitis_B":Hepatitis_B,
         "Measles":Measles,
          "BMI":BMI,
          "Polio":Polio,
          "HIV_AIDS":HIV_AIDS,
          "Alcohol":Alcohol,
          "thinness_1_19_years":thinness_1_19_years,
          "thinness_5_9_years":thinness_5_9_years,
          "Schooling":Schooling,
          "Country":Country,
          "status" : Status,
          "Year": Year
         }
    features=pd.DataFrame(data, index=[0])
    return features

df=InputData()
df=df
st.subheader("Input Parameters:")
st.write(df)
GB=GradientBoostingClassifier()

def Modeling():
    le = LabelEncoder()
    def EncodingCategoricals(columns ,df):
        for c in columns:
            df[c]=le.fit_transform(df[c])
    objects=Data.select_dtypes("float64")
    EncodingCategoricals(Data.columns,Data)
    y=Data['Life_expectancy ']
    X=Data.drop('Life_expectancy ', axis=1)
    sc=StandardScaler()
    X=sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=0)
    GB.fit(X_train,y_train)

st.subheader("Life Expectancy:")
def PredictAns():
        Modeling()
        y_predict=GB.predict(df)
        return y_predict
if( st.sidebar.button("Predict")):
    ans=PredictAns()
    if(ans==0):
        st.write("Congratulations! The life expectancy is :)")
    else:
        st.write("Unfortunately, The selected life expectancy has been denied :(")





