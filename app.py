import streamlit as st
import pandas as pd
import joblib
import random
import os
import json
import urllib.request
import urllib.error



def generate_ai_investor_feedback(
    industry,
    started_in,
    age_group,
    num_founders,
    monthly_sales,
    ask_amount,
    equity,
    has_patent,
    selected_sharks,
    probability,
    valuation_requested
):
    """
    Generate investor-style feedback using Gemini.
    The ML model remains responsible for the funding probability;
    Gemini is used only for qualitative feedback.
    """

    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

    if not api_key:
        return (
            "AI feedback is not configured yet. "
            "Add GEMINI_API_KEY to Streamlit secrets."
        )

    prompt = f"""
You are an experienced startup investor giving feedback after a Shark Tank-style pitch.

Your job is to provide specific, evidence-based feedback using ONLY the information supplied below.
Do not invent revenue growth, market size, customers, competitors, margins, or other facts.

IMPORTANT:
- The machine-learning model has already calculated the funding probability.
- Do NOT change or recalculate that probability.
- Your role is qualitative analysis and investor feedback.
- Be concise and practical.
- Distinguish strengths, concerns, and actionable improvements.
- Do not guarantee that an investor will make an offer.

STARTUP INFORMATION
Industry: {industry}
Startup started in: {started_in}
Founder age group: {age_group}
Number of founders/presenters: {num_founders}
Monthly sales: ₹{monthly_sales:,.0f}
Funding ask: ₹{ask_amount:,.0f}
Equity offered: {equity:.1f}%
Patent: {has_patent}
Today's shark panel: {", ".join(selected_sharks)}

CALCULATED METRICS
Implied valuation: ₹{valuation_requested:,.0f}
PitchWise ML funding probability: {probability * 100:.1f}%

Return the response in exactly this structure:

### Overall Investor View
2-3 sentences.

### Strengths
- 2 or 3 specific strengths based only on the supplied information.

### Key Concerns
- 2 or 3 specific investor concerns based only on the supplied information.

### What I Would Improve
- 3 actionable recommendations for the founder.

### Investor Verdict
One short sentence describing whether the pitch looks compelling, borderline, or weak based on the supplied information. Do not make a guaranteed investment decision.
"""

    # Gemini REST API avoids adding another Python package to the Streamlit app.
    model_name = "gemini-2.5-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 700
        }
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))

        candidates = result.get("candidates", [])
        if not candidates:
            return "Gemini returned no feedback. Please try again."

        parts = candidates[0].get("content", {}).get("parts", [])
        feedback = "".join(
            part.get("text", "") for part in parts
        ).strip()

        return feedback or "Gemini returned empty feedback."

    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            error_body = str(e)

        return (
            f"AI feedback could not be generated (API error {e.code}). "
            f"Please check your Gemini API key/model configuration."
        )

    except Exception as e:
        return (
            "AI feedback could not be generated right now. "
            "Please check your internet connection and Gemini API configuration."
        )


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



    st.subheader("🤖 AI Investor Feedback")

    st.caption(
        "Gemini analyzes the startup information and PitchWise prediction "
        "to provide qualitative investor-style feedback."
    )

    with st.spinner("🧠 AI investor is analyzing your pitch..."):
        ai_feedback = generate_ai_investor_feedback(
            industry=industry,
            started_in=started_in,
            age_group=age_group,
            num_founders=num_founders,
            monthly_sales=monthly_sales,
            ask_amount=ask_amount,
            equity=equity,
            has_patent=has_patent,
            selected_sharks=selected_sharks,
            probability=probability,
            valuation_requested=valuation_requested
        )

    st.markdown(ai_feedback)


st.divider()
st.caption(
    "PitchWise uses a machine-learning model for the funding probability "
    "and Gemini for qualitative investor feedback."
)
