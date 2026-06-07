import streamlit as st
import pandas as pd
import joblib
import random


st.set_page_config(
    page_title="PitchWise",
    page_icon="🚀",
    layout="centered"
)


model = joblib.load("pitchwise_model.pkl")
industry_encoder = joblib.load("industry_encoder.pkl")
age_encoder = joblib.load("age_encoder.pkl")


st.title("🚀 PitchWise")
st.subheader("Startup Funding Prediction Simulator")

st.markdown(
    """
Enter your startup details and see the probability
of receiving funding from investors.
"""
)


industry = st.selectbox(
    "Industry",
    industry_encoder.classes_
)

started_in = st.number_input(
    "Startup Started In",
    min_value=1995,
    max_value=2026,
    value=2022
)

age_group = st.selectbox(
    "Founder Age Group",
    age_encoder.classes_
)

num_founders = st.number_input(
    "Number of Founders",
    min_value=1,
    max_value=10,
    value=1
)

monthly_sales = st.number_input(
    "Monthly Sales (₹)",
    min_value=0.0,
    value=100000.0
)

ask_amount = st.number_input(
    "Funding Ask Amount (₹)",
    min_value=0.0,
    value=1000000.0
)

equity = st.number_input(
    "Equity Offered (%)",
    min_value=1.0,
    max_value=100.0,
    value=10.0
)

has_patent = st.radio(
    "Do you have a patent?",
    ["No", "Yes"]
)


if st.button("Predict Funding Probability"):

  
    yearly_revenue = monthly_sales * 12

    valuation_requested = ask_amount / (equity / 100)
  

    industry_encoded = industry_encoder.transform([industry])[0]
    age_encoded = age_encoder.transform([age_group])[0]

    patent_value = 1 if has_patent == "Yes" else 0



    all_sharks = [
        "Namita",
        "Vineeta",
        "Anupam",
        "Aman",
        "Peyush",
        "Ritesh",
        "Amit"
    ]

    selected_sharks = random.sample(all_sharks, 5)

    st.subheader("🦈 Today's Shark Panel")

    for shark in selected_sharks:
        st.write(f"✓ {shark}")



    input_data = pd.DataFrame([{
        "Industry": industry_encoded,
        "Started in": started_in,
        "Number of Presenters": num_founders,
        "Male Presenters": num_founders,
        "Female Presenters": 0,
        "Couple Presenters": 0,
        "Pitchers Average Age": age_encoded,
        "Yearly Revenue": yearly_revenue,
        "Monthly Sales": monthly_sales,
        "Has Patents": patent_value,
        "Bootstrapped": 0,
        "Original Ask Amount": ask_amount,
        "Original Offered Equity": equity,
        "Valuation Requested": valuation_requested,
        "Namita Present": int("Namita" in selected_sharks),
        "Vineeta Present": int("Vineeta" in selected_sharks),
        "Anupam Present": int("Anupam" in selected_sharks),
        "Aman Present": int("Aman" in selected_sharks),
        "Peyush Present": int("Peyush" in selected_sharks),
        "Ritesh Present": int("Ritesh" in selected_sharks),
        "Amit Present": int("Amit" in selected_sharks),
        "Guest Present": 0
    }])


    probability = model.predict_proba(input_data)[0][1]

    prediction = model.predict(input_data)[0]


    st.divider()

    st.subheader("📊 Funding Analysis")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Funding Chance",
            f"{probability * 100:.1f}%"
        )

    with col2:
        st.metric(
            "PitchWise Score",
            f"{int(probability * 100)}/100"
        )

    with col3:
        st.metric(
            "Implied Valuation",
            f"₹{valuation_requested:,.0f}"
        )


    if probability >= 0.70:
        st.success("🟢 High Funding Potential")

    elif probability >= 0.50:
        st.warning("🟡 Moderate Funding Potential")

    else:
        st.error("🔴 Low Funding Potential")



    positives = []
    risks = []

    if monthly_sales >= 100000:
        positives.append("Strong monthly sales")

    if equity >= 5:
        positives.append("Investor-friendly equity offer")

    if has_patent == "Yes":
        positives.append("Protected intellectual property")

    if valuation_requested > yearly_revenue * 20:
        risks.append("Valuation appears aggressive")

    if monthly_sales < 50000:
        risks.append("Limited sales traction")

    if equity < 2:
        risks.append("Very low equity offered")

    st.subheader("💡 Investor Feedback")

    if positives:
        for item in positives:
            st.success(f"✓ {item}")

    if risks:
        for item in risks:
            st.warning(f"⚠ {item}")

    if not positives and not risks:
        st.info("No major strengths or concerns detected.")
