
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pickle as p
import tensorflow as tf
import streamlit as st

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# --------------------------------------------------
# 2. LOAD MODEL AND PREPROCESSING FILES
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("model.h5")

    with open("scaler.pkl", "rb") as f:
        scaler = p.load(f)

    with open("label_gender.pkl", "rb") as f:
        label_encoder_gender = p.load(f)

    with open("one_hot_geo.pkl", "rb") as f:
        one_hot_encoder_geo = p.load(f)

    return (
        model,
        scaler,
        label_encoder_gender,
        one_hot_encoder_geo
    )


try:
    (
        model,
        scaler,
        label_encoder_gender,
        one_hot_encoder_geo
    ) = load_artifacts()

except Exception as e:
    st.error(
        "Model ya preprocessing files load nahi ho paayi. "
        "Files aur dependencies check karo."
    )
    st.exception(e)
    st.stop()


# --------------------------------------------------
# 3. APP HEADER
# --------------------------------------------------
st.title("📊 Customer Churn Prediction")
st.write(
    "Customer ki details enter karo aur predict karo "
    "ki customer bank chhod sakta hai ya nahi."
)

st.info(
    "Note: Prediction model ke output par based hai. "
    "Ye guaranteed result nahi hai."
)

# --------------------------------------------------
# 4. CUSTOMER INPUTS
# --------------------------------------------------
st.subheader("Customer Details")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox(
            "Geography",
            options=one_hot_encoder_geo.categories_[0].tolist()
        )

        gender = st.selectbox(
            "Gender",
            options=label_encoder_gender.classes_.tolist()
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=0,
            max_value=1000,
            value=650,
            step=1
        )

        tenure = st.slider(
            "Tenure (Years)",
            min_value=0,
            max_value=10,
            value=5
        )

    with col2:
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=60000.0,
            step=1000.0
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0,
            value=50000,
            step=1000
        )

        numof_products = st.selectbox(
            "Number of Products",
            options=[1, 2, 3, 4]
        )

        hascards = st.selectbox(
            "Has Credit Card?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        isactive_member = st.selectbox(
            "Is Active Member?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    submitted = st.form_submit_button(
        "🔍 Predict Churn",
        use_container_width=True
    )


# --------------------------------------------------
# 5. PREPARE INPUT AND PREDICT
# --------------------------------------------------
if submitted:
    try:
        # Create input DataFrame
        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [
                label_encoder_gender.transform([gender])[0]
            ],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [numof_products],
            "HasCrCard": [hascards],
            "IsActiveMember": [isactive_member],
            "EstimatedSalary": [estimated_salary]
        })

        # One-hot encode Geography
        geo_encoded = one_hot_encoder_geo.transform(
            [[geography]]
        )

        # Support both sparse and dense encoder output
        if hasattr(geo_encoded, "toarray"):
            geo_encoded = geo_encoded.toarray()

        geo_columns = (
            one_hot_encoder_geo.get_feature_names_out(
                ["Geography"]
            )
        )

        geo_df = pd.DataFrame(
            geo_encoded,
            columns=geo_columns
        )

        # Combine numerical/encoded input and geography
        input_data = pd.concat(
            [input_data, geo_df],
            axis=1
        )

        # Match exact training feature order
        if hasattr(scaler, "feature_names_in_"):
            expected_columns = list(scaler.feature_names_in_)

            missing_columns = [
                col for col in expected_columns
                if col not in input_data.columns
            ]

            extra_columns = [
                col for col in input_data.columns
                if col not in expected_columns
            ]

            if missing_columns or extra_columns:
                raise ValueError(
                    f"Feature mismatch. Missing: {missing_columns}; "
                    f"Extra: {extra_columns}"
                )

            input_data = input_data[expected_columns]

        # Validate values
        if input_data.isnull().any().any():
            raise ValueError(
                "Input data mein missing values hain."
            )

        if not np.isfinite(
            input_data.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Input data mein invalid numeric values hain."
            )

        # Scale inputs
        input_scaled = scaler.transform(input_data)

        # Predict probability
        prediction = model.predict(
            input_scaled,
            verbose=0
        )

        churn_probability = float(
            np.asarray(prediction).reshape(-1)[0]
        )

        if not 0 <= churn_probability <= 1:
            raise ValueError(
                "Model ne valid probability return nahi ki."
            )

        # --------------------------------------------------
        # 6. DISPLAY RESULTS
        # --------------------------------------------------
        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            label="Churn Probability",
            value=f"{churn_probability:.2%}"
        )

        st.progress(churn_probability)

        if churn_probability >= 0.5:
            st.error(
                "⚠️ Customer is likely to leave."
            )
            st.write(
                "Suggested action: Customer feedback aur "
                "retention offers consider karein."
            )
        else:
            st.success(
                "✅ Customer is likely to stay."
            )
            st.write(
                "Suggested action: Customer engagement "
                "maintain rakhein."
            )

        st.caption(
            "Prediction 0.5 threshold par based hai. "
            "Actual performance model training aur evaluation "
            "par depend karti hai."
        )

        with st.expander("View processed input"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(
            "Prediction complete nahi ho paayi. "
            "Feature names, encoders, scaler aur model "
            "compatibility check karo."
        )
        st.exception(e)


# --------------------------------------------------
# 7. ABOUT PROJECT
# --------------------------------------------------
with st.expander("About this project"):
    st.markdown("""
    **Project:** Customer Churn Prediction

    **Technology used:**
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - TensorFlow / Keras
    - Streamlit

    **Workflow:**
    1. Customer details input li jaati hain.
    2. Categorical values encode ki jaati hain.
    3. Input features scale kiye jaate hain.
    4. Trained neural network prediction karta hai.
    5. Churn probability aur result display hota hai.
    """)
