import os
import random
import joblib
import pandas as pd
import streamlit as st

try:
    import google.generativeai as genai
except ImportError:
    genai = None

st.set_page_config(page_title="PitchWise", page_icon="🚀", layout="centered")

# Final Logistic Regression pipeline: preprocessing + model
model = joblib.load("pitchwise_model.pkl")

st.title("🚀 PitchWise")
st.subheader("Startup Funding Prediction Simulator")
st.markdown(
    "Enter your startup details and see the estimated probability "
    "of receiving an offer from investors."
)

INDUSTRIES = ['Agriculture', 'Animal/Pets', 'Beauty/Fashion', 'Business Services', 'Children/Education', 'Electronics', 'Entertainment', 'Fitness/Sports/Outdoors', 'Food and Beverage', 'Green/CleanTech', 'Hardware', 'Lifestyle/Home', 'Liquor/Alcohol', 'Manufacturing', 'Medical/Health', 'Others', 'Technology/Software', 'Vehicles/Electrical Vehicles']
AGE_GROUPS = ['Middle', 'Old', 'Young']

industry = st.selectbox("Industry", INDUSTRIES)

started_in = st.number_input(
    "Startup Started In", min_value=1990, max_value=2026, value=2022, step=1
)

age_group = st.selectbox("Founder Age Group", AGE_GROUPS)

num_founders = st.number_input(
    "Number of Founders / Presenters",
    min_value=1, max_value=6, value=2, step=1
)

female_presenters = st.number_input(
    "Female Presenters",
    min_value=0, max_value=int(num_founders), value=0, step=1
)

yearly_revenue_rupees = st.number_input(
    "Yearly Revenue (₹)",
    min_value=0.0, value=1200000.0, step=50000.0
)

net_margin = st.number_input(
    "Net Margin (%)",
    min_value=-100.0, max_value=100.0, value=10.0, step=1.0
)

ebitda_rupees = st.number_input(
    "EBITDA (₹)",
    min_value=-10000000.0, value=100000.0, step=10000.0
)

ask_amount_rupees = st.number_input(
    "Funding Ask Amount (₹)",
    min_value=0.0, value=1000000.0, step=50000.0
)

equity = st.number_input(
    "Equity Offered (%)",
    min_value=0.1, max_value=100.0, value=10.0, step=0.5
)

if st.button("Predict Funding Probability", type="primary"):

    # The Shark Tank dataset stores monetary values in lakhs.
    # The UI accepts rupees, so convert rupees -> lakhs.
    yearly_revenue = yearly_revenue_rupees / 100000.0
    ebitda = ebitda_rupees / 100000.0
    ask_amount = ask_amount_rupees / 100000.0

    # Current pitch year used for the deployed simulator.
    pitch_year = 2026
    startup_age = max(0, pitch_year - int(started_in))

    selected_sharks = random.sample(
        ["Namita", "Vineeta", "Anupam", "Aman", "Peyush", "Ritesh", "Amit"],
        5
    )

    st.subheader("🦈 Today's Shark Panel")
    for shark in selected_sharks:
        st.write(f"✓ {shark}")

    # EXACTLY the 10 features used by the final model.
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

    probability = float(model.predict_proba(input_data)[0][1])
    threshold = 0.6570
    valuation_requested = ask_amount_rupees / (equity / 100.0)

    st.divider()
    st.subheader("📊 Funding Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Funding Chance", f"{probability * 100:.1f}%")

    with col2:
        st.metric("PitchWise Score", f"{int(round(probability * 100))}/100")

    with col3:
        st.metric("Implied Valuation", f"₹{valuation_requested:,.0f}")

    if probability >= threshold:
        st.success("🟢 High Funding Potential")
    elif probability >= 0.50:
        st.warning("🟡 Moderate Funding Potential")
    else:
        st.error("🔴 Low Funding Potential")

    # =========================================================
    # AI INVESTOR FEEDBACK — GEMINI
    # =========================================================
    st.divider()
    st.subheader("💡 AI Investor Feedback")
    st.caption(
        "Gemini analyzes the startup information together with the "
        "PitchWise prediction to provide qualitative investor-style feedback."
    )

    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

    if genai is None:
        st.error(
            "Gemini package is not installed. Add "
            "google-generativeai to requirements.txt."
        )
    elif not api_key:
        st.warning(
            "Gemini API key is not configured. Add GEMINI_API_KEY "
            "to Streamlit Secrets."
        )
    else:
        try:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel("gemini-2.5-flash")

            prompt = f"""
You are a concise startup investor.

Review this startup using ONLY these facts:
Industry: {industry}
Startup age: {startup_age} years
Presenters: {num_founders}
Female presenters: {female_presenters}
Founder age group: {age_group}
Yearly revenue: ₹{yearly_revenue_rupees:,.0f}
Net margin: {net_margin}%
EBITDA: ₹{ebitda_rupees:,.0f}
Ask: ₹{ask_amount_rupees:,.0f}
Equity: {equity}%
Implied valuation: ₹{valuation_requested:,.0f}
ML funding probability: {probability * 100:.1f}%

Write COMPLETE feedback in 80 words or fewer.

Use exactly these four headings:
### Overall Investor View
Write exactly 1 sentence.

### Key Strengths
Write exactly 2 short bullet points.

### Key Risks
Write exactly 2 short bullet points.

### Investor Recommendation
Write exactly 1 sentence.

Do not invent facts. Do not stop mid-sentence.
"""

            response = gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 300
                }
            )

            feedback = getattr(response, "text", None)

            # Retry once if Gemini unexpectedly returns an incomplete response.
            if feedback:
                stripped = feedback.strip()
                if not stripped.endswith((".", "!", "?", "`", ")")):
                    retry_prompt = prompt + """
Return the complete answer again in under 60 words.
Every sentence must be complete. Do not truncate the response.
"""
                    retry = gemini_model.generate_content(
                        retry_prompt,
                        generation_config={
                            "temperature": 0.1,
                            "max_output_tokens": 220
                        }
                    )
                    retry_text = getattr(retry, "text", None)
                    if retry_text:
                        feedback = retry_text

            if feedback:
                st.markdown(feedback)
            else:
                st.warning("Gemini returned an empty response.")

        except Exception as e:
            st.error(f"Gemini feedback could not be generated: {e}")

st.caption(
    "PitchWise uses Logistic Regression for funding prediction "
    "and Gemini for qualitative investor feedback."
)
