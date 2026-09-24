import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ChurnIQ | Customer Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# PATHS + MODEL
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_pipeline.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        f"Model file not found at:\n\n{MODEL_PATH}\n\n"
        "Make sure churn_pipeline.pkl is inside the models folder."
    )
    st.stop()


model = load_model()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        padding: 32px 36px;
        border-radius: 20px;
        margin-bottom: 28px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.22);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        color: white;
        margin: 0;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #cbd5e1;
        margin-top: 8px;
        margin-bottom: 0;
    }

    .section-heading {
        font-size: 23px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .section-subtitle {
        color: #94a3b8;
        margin-bottom: 18px;
    }

    .prediction-card {
        padding: 26px;
        border-radius: 18px;
        border: 1px solid rgba(148,163,184,0.25);
        background: rgba(30,41,59,0.35);
        margin-bottom: 18px;
    }

    .risk-label {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .probability-value {
        font-size: 48px;
        font-weight: 800;
        margin: 5px 0;
    }

    .muted-text {
        color: #94a3b8;
        font-size: 14px;
    }

    .feature-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(148,163,184,0.18);
        background: rgba(30,41,59,0.25);
        margin-bottom: 10px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        min-height: 3.2rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 16px;
    }

    div[data-testid="stMetric"] {
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(148,163,184,0.18);
    }

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    '<div class="hero-card">'
    '<div class="hero-title">📊 ChurnIQ</div>'
    '<div class="hero-subtitle">'
    'Machine Learning Powered Customer Churn Prediction & Retention Intelligence System'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MAIN LAYOUT
# =========================================================

input_col, result_col = st.columns(
    [1.3, 1],
    gap="large"
)


# =========================================================
# LEFT SIDE - INPUTS
# =========================================================

with input_col:

    st.markdown(
        '<div class="section-heading">Customer Profile</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Enter customer information and let the ML model estimate churn risk.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("customer_form"):

        # -------------------------------------------------
        # PERSONAL INFORMATION
        # -------------------------------------------------

        st.subheader("👤 Customer Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

        with c2:
            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        with c3:
            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

        c1, c2, c3 = st.columns(3)

        with c1:
            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

        with c2:
            tenure = st.slider(
                "Tenure (Months)",
                min_value=0,
                max_value=72,
                value=12
            )

        with c3:
            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )


        st.write("")


        # -------------------------------------------------
        # SERVICE DETAILS
        # -------------------------------------------------

        with st.expander(
            "🌐 Service Details",
            expanded=False
        ):

            c1, c2, c3 = st.columns(3)

            with c1:
                phone_service = st.selectbox(
                    "Phone Service",
                    ["Yes", "No"]
                )

            with c2:
                multiple_lines = st.selectbox(
                    "Multiple Lines",
                    [
                        "No",
                        "Yes",
                        "No phone service"
                    ]
                )

            with c3:
                internet_service = st.selectbox(
                    "Internet Service",
                    [
                        "DSL",
                        "Fiber optic",
                        "No"
                    ]
                )

            c1, c2, c3 = st.columns(3)

            with c1:
                online_security = st.selectbox(
                    "Online Security",
                    [
                        "Yes",
                        "No",
                        "No internet service"
                    ]
                )

            with c2:
                online_backup = st.selectbox(
                    "Online Backup",
                    [
                        "Yes",
                        "No",
                        "No internet service"
                    ]
                )

            with c3:
                device_protection = st.selectbox(
                    "Device Protection",
                    [
                        "Yes",
                        "No",
                        "No internet service"
                    ]
                )

            c1, c2, c3 = st.columns(3)

            with c1:
                tech_support = st.selectbox(
                    "Tech Support",
                    [
                        "Yes",
                        "No",
                        "No internet service"
                    ]
                )

            with c2:
                streaming_tv = st.selectbox(
                    "Streaming TV",
                    [
                        "Yes",
                        "No",
                        "No internet service"
                    ]
                )

            with c3:
                streaming_movies = st.selectbox(
                    "Streaming Movies",
                    [
                        "Yes",
                        "No",
                        "No internet service"
                    ]
                )


        st.write("")


        # -------------------------------------------------
        # BILLING
        # -------------------------------------------------

        st.subheader("💳 Contract & Billing")

        c1, c2 = st.columns(2)

        with c1:
            contract = st.selectbox(
                "Contract Type",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

        with c2:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        c1, c2 = st.columns(2)

        with c1:
            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0
            )

        with c2:
            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=700.0,
                step=10.0
            )


        st.write("")

        predict_button = st.form_submit_button(
            "🔍 Analyse Customer",
            use_container_width=True
        )


# =========================================================
# CREATE CUSTOMER DATAFRAME
# =========================================================

customer = pd.DataFrame(
    [{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }]
)


# =========================================================
# RIGHT SIDE - RESULTS
# =========================================================

with result_col:

    st.markdown(
        '<div class="section-heading">Prediction Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Model prediction, churn probability and retention insights.'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # BEFORE PREDICTION
    # -----------------------------------------------------

    if not predict_button:

        st.info(
            "Enter customer information and click "
            "**Analyse Customer** to generate the prediction."
        )

        st.write("")

        st.subheader("What ChurnIQ Provides")

        st.markdown(
            """
- ML-based churn prediction
- Churn probability score
- Risk classification
- Customer profile summary
- Risk indicators
- Retention suggestions
"""
        )

        st.write("")

        st.markdown(
            '<div class="feature-card">'
            '<b>How it works</b><br><br>'
            'Customer Data → Preprocessing → ML Pipeline → '
            'Prediction → Business Insights'
            '</div>',
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # AFTER PREDICTION
    # -----------------------------------------------------

    else:

        prediction = model.predict(customer)[0]

        probability = model.predict_proba(
            customer
        )[0][1]


        # -------------------------------------------------
        # RISK CLASSIFICATION
        # -------------------------------------------------

        if probability >= 0.70:
            risk_level = "HIGH RISK"
            risk_icon = "🔴"

        elif probability >= 0.40:
            risk_level = "MODERATE RISK"
            risk_icon = "🟠"

        else:
            risk_level = "LOW RISK"
            risk_icon = "🟢"


        # -------------------------------------------------
        # MAIN RESULT CARD
        # -------------------------------------------------

        st.markdown(
            '<div class="prediction-card">'
            f'<div class="risk-label">{risk_icon} {risk_level}</div>'
            f'<div class="probability-value">{probability:.1%}</div>'
            '<div class="muted-text">Estimated churn probability</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.progress(
            min(float(probability), 1.0)
        )

        st.write("")


        # -------------------------------------------------
        # FINAL MODEL PREDICTION
        # -------------------------------------------------

        if prediction == 1:

            st.error(
                "⚠️ The model predicts that this customer "
                "is likely to churn."
            )

        else:

            st.success(
                "✅ The model predicts that this customer "
                "is likely to stay."
            )


        st.divider()


        # -------------------------------------------------
        # CUSTOMER SUMMARY
        # -------------------------------------------------

        st.subheader("📋 Customer Summary")

        m1, m2 = st.columns(2)

        with m1:

            st.metric(
                "Tenure",
                f"{tenure} months"
            )

            st.metric(
                "Contract",
                contract
            )

        with m2:

            st.metric(
                "Monthly Charges",
                f"${monthly_charges:.2f}"
            )

            st.metric(
                "Internet Service",
                internet_service
            )


        st.divider()


        # -------------------------------------------------
        # RISK INDICATORS
        # -------------------------------------------------

        st.subheader("🔎 Risk Indicators")

        indicators = []

        if contract == "Month-to-month":
            indicators.append(
                "Month-to-month contract"
            )

        if tenure < 12:
            indicators.append(
                "Low customer tenure"
            )

        if tech_support == "No":
            indicators.append(
                "No technical support service"
            )

        if online_security == "No":
            indicators.append(
                "No online security service"
            )

        if payment_method == "Electronic check":
            indicators.append(
                "Electronic check payment method"
            )

        if monthly_charges > 80:
            indicators.append(
                "Relatively high monthly charges"
            )


        if indicators:

            for indicator in indicators:

                st.markdown(
                    f"• {indicator}"
                )

        else:

            st.success(
                "No major rule-based risk indicators detected."
            )


        st.caption(
            "Risk indicators are rule-based observations from "
            "the customer's profile. They are not direct explanations "
            "of the machine learning model."
        )


        st.divider()


        # -------------------------------------------------
        # RETENTION SUGGESTIONS
        # -------------------------------------------------

        st.subheader("💡 Retention Suggestions")

        suggestions = []

        if contract == "Month-to-month":

            suggestions.append(
                "Offer incentives for switching to a longer-term contract."
            )

        if tech_support == "No":

            suggestions.append(
                "Offer technical support assistance or a free trial."
            )

        if monthly_charges > 80:

            suggestions.append(
                "Review personalised pricing or bundled service options."
            )

        if tenure < 12:

            suggestions.append(
                "Provide an early-stage loyalty or engagement offer."
            )

        if payment_method == "Electronic check":

            suggestions.append(
                "Promote convenient automatic payment options."
            )


        if suggestions:

            for suggestion in suggestions:

                st.markdown(
                    f"✓ {suggestion}"
                )

        else:

            st.success(
                "The customer currently shows relatively few "
                "obvious retention-risk indicators."
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

f1, f2 = st.columns([3, 1])

with f1:
    st.caption(
        "ChurnIQ • Machine Learning Customer Churn Prediction System"
    )

with f2:
    st.caption(
        "Built with Python • Scikit-learn • Streamlit"
    )