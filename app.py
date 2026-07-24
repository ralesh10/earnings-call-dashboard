import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Earnings Call Analyzer", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('dashboard_data.csv')

@st.cache_resource
def load_model():
    return joblib.load('earnings_model.pkl')

df = load_data()
model = load_model()

feature_cols = joblib.load('feature_cols.pkl')

st.title("📈 AI Earnings Call Analyzer")
st.caption("Predicting Stock Outperformance Using NLP")

# ---- SIDEBAR ----
st.sidebar.header("Select Company")
ticker = st.sidebar.selectbox("Company", sorted(df['symbol'].unique()))

company_df = df[df['symbol'] == ticker].sort_values('call_datetime')
quarter_options = company_df['call_datetime'].tolist()
st.sidebar.header("Select Quarter")
selected_date = st.sidebar.selectbox("Quarter", quarter_options)

analyze = st.sidebar.button("Analyze", type="primary")

# ---- MAIN PAGE ----
if analyze:
    row = company_df[company_df['call_datetime'] == selected_date].iloc[0]
    X = row[feature_cols].values.reshape(1, -1)
    prob = model.predict_proba(X)[0, 1]
    prediction = "OUTPERFORM" if prob >= 0.5 else "UNDERPERFORM"
    emoji = "🟢" if prediction == "OUTPERFORM" else "🔴"

    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction", f"{emoji} {prediction}")
    col2.metric("Confidence", f"{prob if prediction=='OUTPERFORM' else 1-prob:.0%}")
    col3.metric("Expected Abnormal Return", "Positive" if prediction == "OUTPERFORM" else "Negative")

    st.divider()
    st.subheader("📊 Model Insights")

    ic1, ic2 = st.columns(2)
    ic1.metric("Sentiment Score", f"{row['qa_pos']:.2f}")
    ic2.metric("Historical Tone Difference", f"{row['sentiment_mismatch_pos']:+.2f}")

    st.write("**Prediction Probability**")
    st.write(f"Outperform {prob:.0%}")
    st.progress(prob)
    st.write(f"Underperform {1-prob:.0%}")
    st.progress(1 - prob)

    st.divider()
    st.subheader("📈 Feature Importance")
    if hasattr(model, 'coef_'):
        importances = pd.Series(abs(model.coef_[0]), index=feature_cols).sort_values(ascending=False)
    else:
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3))
    importances.head(5).plot(kind='barh', ax=ax)
    ax.invert_yaxis()
    st.pyplot(fig)

    st.divider()
    st.subheader("Actual Outcome")
    if 'abnormal_return_5d' in row and pd.notnull(row['abnormal_return_5d']):
        actual = "✅ Outperformed Market" if row['abnormal_return_5d'] > 0 else "❌ Underperformed Market"
        st.write(actual)
    else:
        st.write("Outcome not yet available (recent/future call)")

    st.caption("⚠️ Model accuracy: 55.7%, AUC 0.63 on historical test data. Not financial advice.")
else:
    st.info("👈 Select a company and quarter, then click Analyze")
