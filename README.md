# 🚀 PitchWise

PitchWise is an AI-powered startup funding prediction simulator inspired by Shark Tank India.

The application uses a machine learning model trained on historical startup pitch data to estimate a startup's funding potential, calculate valuation, generate a realistic investor panel, and provide investor-style feedback.

## 🌐 Live Demo

Try PitchWise here:

[🚀 Launch PitchWise](https://pitchwise-startup-pitch-simulator-yug.streamlit.app/)

---

## 🎯 Features

- Predict startup funding potential using Machine Learning
- Simulate a realistic Shark Tank investor panel
- Calculate implied startup valuation
- Analyze startup strengths and risks
- Generate investor-style feedback
- Interactive Streamlit web application

---

## 🧠 Machine Learning

The model was trained using historical startup pitch data and evaluates factors such as:

- Industry
- Startup age
- Founder count
- Founder age group
- Revenue and sales
- Funding ask amount
- Equity offered
- Patent ownership
- Investor panel composition

The final model uses a Random Forest Classifier built with Scikit-Learn.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib

---

## 📂 Project Structure

```text
PitchWise/
│
├── app.py
├── pitchwise_model.pkl
├── industry_encoder.pkl
├── age_encoder.pkl
├── requirements.txt
└── README.md
```

---

## ▶️ Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📈 Example Workflow

1. Enter startup details
2. Generate investor panel
3. Predict funding potential
4. View valuation estimate
5. Receive investor-style feedback

---

## 👨‍💻 Author

Yug Bagrodia
