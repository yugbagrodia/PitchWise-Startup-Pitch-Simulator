import streamlit as st
import pandas as pd
import joblib
import random
from datetime import datetime

st.set_page_config(
    page_title="PitchWise",
    page_icon="🚀",
    layout="centered"
)

model = joblib.load("pitchwise_model.pkl")

st.title("🚀 PitchWise")
st.subheader("Startup Funding Prediction Simulator")

st.markdown(
    """
    Enter your startup details and see the estimated probability
    of receiving an offer from investors.
    """
)

industry = st.selectbox(
    "Industry",
    [
        "Food and Beverage",
        "Vehicles/Electrical Vehicles",
        "Beauty/Fashion",
        "Children/Education",
        "Agriculture",
        "Medical/Health",
        "Manufacturing",
        "Technology/Software",
        "Green/CleanTech",
        "Electronics",
        "Animal/Pets",
        "Business Services",
        "Hardware",
        "Fitness/Sports/Outdoors",
        "Entertainment",
        "Other"
    ]
)

started_in = st.number_input(
    "Startup Started In",
    min_value=1990,
    max_value=datetime.now().year,
    value=2022,
    step=1
)

age_group = st.selectbox(
    "Founder Age Group",
    ["Young", "Middle", "Old"]
)

num_founders = st.number_input(
    "Number of Founders / Presenters",
    min_value=1,
    max_value=10,
    value=1,
    step=1
)

female_presenters = st.number_input(
    "Female Presenters",
    min_value=0,
    max_value=num_founders,
    value=0,
    step=1
)

yearly_revenue = st.number_input(
    "Yearly Revenue (₹)",
    min_value=0.0,
    value=1200000.0,
    step=10000.0
)

net_margin = st.number_input(
    "Net Margin (%)",
    min_value=-100.0,
    max_value=100.0,
    value=10.0,
    step=1.0
)

ebitda = st.number_input(
    "EBITDA (₹)",
    value=100000.0,
    step=10000.0
)

ask_amount = st.number_input(
    "Funding Ask Amount (₹)",
    min_value=0.0,
    value=1000000.0,
    step=50000.0
)

equity = st.number_input(
    "Equity Offered (%)",
    min_value=0.1,
    max_value=100.0,
    value=10.0,
    step=0.5
)

# The deployed simulator represents a current pitch.
pitch_year = datetime.now().year
startup_age = max(0, pitch_year - int(started_in))

if st.button("Predict Funding Probability", type="primary"):

    selected_sharks = random.sample(
        ["Namita", "Vineeta", "Anupam", "Aman", "Peyush", "Ritesh", "Amit"],
        5
    )

    st.subheader("🦈 Today's Shark Panel")
    for shark in selected_sharks:
        st.write(f"✓ {shark}")

    input_data = pd.DataFrame([{
        "Industry": industry,
        "Startup Age at Pitch": startup_age,
        "Number of Presenters": num_founders,
        "Female Presenters": female_presenters,
        "Pitchers Average Age": age_group,
        "Yearly Revenue": yearly_revenue,
        "Net Margin": net_margin,
        "EBITDA": ebitda,
        "Original Ask Amount": ask_amount,
        "Original Offered Equity": equity
    }])

    probability = model.predict_proba(input_data)[0][1]

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
        valuation_requested = ask_amount / (equity / 100)
        st.metric(
            "Implied Valuation",
            f"₹{valuation_requested:,.0f}"
        )

    # Keep the notebook's 0.657 probability threshold for the
    # custom funding classification.
    if probability >= 0.657:
        st.success("🟢 High Funding Potential")
    elif probability >= 0.50:
        st.warning("🟡 Moderate Funding Potential")
    else:
        st.error("🔴 Low Funding Potential")

    st.subheader("💡 Investor Feedback")

    positives = []
    risks = []

    if yearly_revenue >= 1000000:
        positives.append("Strong yearly revenue")
    if net_margin > 0:
        positives.append("Positive net margin")
    if ebitda > 0:
        positives.append("Positive EBITDA")
    if equity >= 5:
        positives.append("Investor-friendly equity offer")

    if net_margin < 0:
        risks.append("Negative net margin")
    if ebitda < 0:
        risks.append("Negative EBITDA")
    if yearly_revenue < 500000:
        risks.append("Limited revenue traction")
    if equity < 2:
        risks.append("Very low equity offered")

    if positives:
        for item in positives:
            st.success(f"✓ {item}")

    if risks:
        for item in risks:
            st.warning(f"⚠ {item}")

    if not positives and not risks:
        st.info("No major strengths or concerns detected.")
