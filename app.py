import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Accident Risk Prediction", layout="wide")

# =========================
# LOAD MODEL
# =========================
import gdown
import os

MODEL_PATH = "risk_prediction_model.pkl"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1tttbxBg_JKuuZTKowXHIuvAyZxTjKONW"
    gdown.download(url, MODEL_PATH, quiet=False)

model=pickle.load(open(MODEL_PATH, "rb"))
@st.cache_data
def load_model():
    return model
model = load_model()

# =========================
# LOAD & FIX TABLES
# =========================
def load_table(file):
    try:
        df = pd.read_csv(file)

        # If everything loaded in ONE column → fix it
        if df.shape[1] == 1:
            df = df[df.columns[0]].str.split(",", expand=True)
            df.columns = df.iloc[0]
            df = df[1:]

        # Set index
        if "Road_Type" in df.columns:
            df.set_index("Road_Type", inplace=True)

        # Convert to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Drop empty rows
        df = df.dropna(how="all")

        return df

    except Exception as e:
        st.error(f"Error loading {file}: {e}")
        return pd.DataFrame()

risk_table_scaled = load_table("risk_table_scaled.csv")
risk_table = load_table("risk_table.csv")

# =========================
# SIDEBAR
# =========================
page = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Data Analysis Dashboard", "Risk Prediction Tool", "Project Utility"]
)

# =========================
# HOME
# =========================
if page == "Home":

    st.title("🚗 Accident Risk Prediction System")

    st.header("Project Overview")
    st.write("""
This system predicts accident risk based on road and environmental conditions 
using machine learning techniques applied to real-world data.
""")

    st.subheader("Dataset Information")
    st.write("""
• Dataset: US Accidents Dataset (Kaggle)  
• Sample Size: ~200,000 records  
• Features Used:
    - Visibility
    - Wind Speed
    - Temperature
    - Humidity
    - Pressure
    - Night Condition
    - Weather Condition
""")

    st.subheader("Techniques Used")
    st.write("""
• Data Cleaning & Preprocessing  
• Feature Engineering  
• Random Forest Classifier  
• Data Visualization & Statistical Analysis  
""")

    st.subheader("Project Goal")
    st.write("""
• Predict accident risk probability  
• Identify key contributing factors  
• Provide safety recommendations  
• Improve road safety awareness  
""")

# =========================
# DASHBOARD
# =========================
elif page == "Data Analysis Dashboard":

    st.title("📊 Data Analysis Dashboard")
    url="https://drive.google.com/uc?export=download&id=1LdXREveLbsn1GO2h8W4T7lL2XGH_rkG7"
    df = pd.read_csv(url)

    @st.cache_data
    def load_data():
        url = "https://drive.google.com/file/d/1LdXREveLbsn1GO2h8W4T7lL2XGH_rkG7/view?usp=drivesdk"
        return pd.read_csv(url)

    # Correlation
    st.subheader("Correlation Heatmap")
    corr = df[['Severity','Visibility(mi)','Wind_Speed(mph)',
               'Temperature(F)','Humidity(%)','Pressure(in)']].corr()

    fig1, ax1 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

    # Accidents by Hour
    st.subheader("Accidents by Hour")
    hour_counts = df['Hour'].value_counts().sort_index()

    fig2, ax2 = plt.subplots()
    ax2.plot(hour_counts.index, hour_counts.values)
    st.pyplot(fig2)

    # Road type risk
    st.subheader("Risk by Road Type")
    road_risk = df.groupby("Road_Type")["Severity"].mean()

    fig3, ax3 = plt.subplots()
    road_risk.plot(kind="bar", ax=ax3)
    st.pyplot(fig3)

    # Boxplot
    st.subheader("Severity Comparison Across Road Types")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x='Road_Type', y='Severity', data=df, ax=ax4)
    st.pyplot(fig4)

    # Heatmap (fixed)
    st.subheader("Risk Factor Contribution by Road Type")

    if risk_table_scaled.empty:
        st.warning("⚠️ Risk table not loaded properly")
        st.write(risk_table_scaled)
    else:
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        sns.heatmap(risk_table_scaled, annot=True, cmap="YlOrRd", linewidths=0.5, ax=ax5)
        st.pyplot(fig5)

    # ---------- Feature Importance ----------
    st.subheader("Feature Importance")
    try:
        importances = model.feature_importances_
        features = [
            'Visibility', 'Wind Speed', 'Temperature',
            'Humidity', 'Pressure', 'Night', 'Bad Weather'
        ]
        fig_imp, ax_imp = plt.subplots()
        ax_imp.barh(features, importances)
        ax_imp.set_xlabel("Importance Score")
        st.pyplot(fig_imp)
    except:
        st.warning("Feature importance not available")

# =========================
# PREDICTION TOOL
# =========================
elif page == "Risk Prediction Tool":

    st.title("⚠️ Accident Risk Prediction Tool")

    # Inputs
    visibility = st.slider("Visibility (miles)", 0.0, 10.0, 5.0)
    wind_speed = st.slider("Wind Speed (mph)", 0.0, 50.0, 10.0)
    temperature = st.slider("Temperature (F)", -10.0, 110.0, 70.0)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    pressure = st.slider("Pressure (in)", 25.0, 35.0, 30.0)

    night = st.selectbox("Night Condition", [0, 1])
    bad_weather = st.selectbox("Bad Weather", [0, 1])

    if st.button("Predict Risk"):

        # Create input dataframe (must match training features)
        input_df = pd.DataFrame([{
            'Visibility(mi)': visibility,
            'Wind_Speed(mph)': wind_speed,
            'Temperature(F)': temperature,
            'Humidity(%)': humidity,
            'Pressure(in)': pressure,
            'Night_Flag': night,
            'Bad_Weather_Flag': bad_weather
        }])

        # ---------- Input Summary ----------
        st.subheader("Input Summary")
        st.write(f"""
                 - Visibility: {visibility}
                 - Wind Speed: {wind_speed}
                 - Temperature: {temperature}
                 - Humidity: {humidity}
                 - Pressure: {pressure}
                 - Night: {night}
                 - Bad Weather: {bad_weather}""")

        # Model prediction
        prediction = model.predict_proba(input_df)
        risk_percent = prediction[0][1] * 100

        # Light logical correction (controlled)
        adjustment = 0

        if bad_weather == 1:
            adjustment += 5
        if night == 1:
            adjustment += 3
        if visibility < 2:
            adjustment += 5
        if wind_speed > 30:
            adjustment += 3

        risk_percent = min(risk_percent + adjustment, 100)

        # Display result
        st.subheader(f"🚨 Risk Score: {risk_percent:.2f}%")

        # Risk level classification
        if risk_percent < 35:
            st.success("Low Risk")
        elif risk_percent < 60:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")

        # ---------- Model Insight ----------
        st.subheader("Model Insight")
        if risk_percent > 60:
            st.write("⚠️ High risk due to multiple environmental stress factors.")
        elif risk_percent > 35:
            st.write("⚠️ Moderate risk due to some unfavorable conditions.")
        else:
            st.write("✅ Conditions are relatively safe with low contributing risk factors.")

        # ---------- Causes ----------
        st.subheader("Possible Causes")

        causes = []
        if visibility < 3:
            causes.append("Low visibility")
        if wind_speed > 25:
            causes.append("High wind speed")
        if humidity > 80:
            causes.append("High humidity")
        if bad_weather == 1:
            causes.append("Bad weather conditions")
        if night == 1:
            causes.append("Night driving")

        if causes:
            for c in causes:
                st.write("•", c)
        else:
            st.write("Conditions are relatively stable.")

        # ---------- Precautions ----------
        st.subheader("Safety Precautions")

        precautions = []
        if visibility < 3:
            precautions.append("Use fog lights and reduce speed")
        if wind_speed > 25:
            precautions.append("Maintain control of steering")
        if bad_weather == 1:
            precautions.append("Avoid sudden braking")
        if night == 1:
            precautions.append("Use proper headlights")

        precautions.append("Maintain safe distance")
        precautions.append("Follow traffic rules")

        for p in precautions:
            st.write("•", p)

        st.subheader("Conclusion")
        st.write("""This prediction combines machine learning insights with environmental analysis 
                 to estimate accident risk and improve safety awareness.""")
        
# =========================
# PROJECT UTILITY
# =========================
elif page == "Project Utility":

    st.title("📌 Project Utility")

    st.write("This system helps users understand accident risk.")

    st.subheader("Applications")
    st.write("• Helps drivers make safer decisions")
    st.write("• Improves road safety awareness")
    st.write("• Assists traffic authorities")
    st.write("• Demonstrates real-world ML application")
