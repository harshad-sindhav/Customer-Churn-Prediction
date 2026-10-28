# Customer Churn Prediction using Deep Learning

## 📌 Project Overview

This project predicts whether a bank customer is likely to leave the bank (churn) using a Deep Learning model built with TensorFlow and Keras.

The application takes customer details as input, applies the required preprocessing steps, and displays the predicted churn probability through an interactive Streamlit web application.

## 🎯 Project Objectives

- Predict whether a customer is likely to leave the bank.
- Apply categorical encoding and feature scaling.
- Use a trained neural network for binary classification.
- Build an interactive web application using Streamlit.
- Understand the end-to-end Machine Learning workflow.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Jupyter Notebook
- Pickle

## 📂 Project Structure

```text
NLP/
├── Churn_Modelling.csv
├── LICENSE
├── README.md
├── algo.ipynb
├── app.py
├── label_gender.pkl
├── model.h5
├── one_hot_geo.pkl
├── prediction.ipynb
├── requirements.txt
└── scaler.pkl
```

## 📊 Dataset

The project uses `Churn_Modelling.csv`, which contains customer information and a churn target.

Customer features include:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Account Balance
- Number of Products
- Credit Card Status
- Active Membership Status
- Estimated Salary

The target variable is `Exited`, which indicates whether a customer left the bank.

## ⚙️ How It Works

1. Customer information is collected through the application.
2. Gender is transformed using a saved Label Encoder.
3. Geography is transformed using a saved One-Hot Encoder.
4. Input features are scaled using a saved Standard Scaler.
5. The trained TensorFlow/Keras model predicts the churn probability.
6. The application displays whether the customer is likely to leave or stay.

## 📁 Model and Preprocessing Files

| File | Purpose |
|---|---|
| `model.h5` | Saved trained neural network |
| `scaler.pkl` | Saved feature scaler |
| `label_gender.pkl` | Saved gender encoder |
| `one_hot_geo.pkl` | Saved geography encoder |
| `algo.ipynb` | Model development notebook |
| `prediction.ipynb` | Prediction and preprocessing notebook |
| `app.py` | Streamlit application |
| `requirements.txt` | Python dependencies |

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/harshad-sindhav/NLP.git
```

### 2. Open the project folder

```bash
cd NLP
```

### 3. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application should open in your browser. If it does not, use the local URL displayed in the terminal.

## 🚀 Deploy on Streamlit Community Cloud

1. Push the project files to your GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using GitHub.
4. Create a new app and select the `harshad-sindhav/NLP` repository.
5. Select the `main` branch.
6. Set the main file path to `app.py`.
7. Click **Deploy**.

Make sure `requirements.txt` contains compatible versions of the required libraries and all model and preprocessing files are available in the repository.

## 📈 Prediction Output

The application displays a churn probability and classifies the customer using the configured probability threshold.

- **Likely to Leave:** Predicted churn probability meets or exceeds the configured threshold.
- **Likely to Stay:** Predicted churn probability is below the configured threshold.

Predictions are estimates and are not guaranteed outcomes.

## 🔮 Future Improvements

- Add model evaluation metrics such as precision, recall, F1-score, and ROC-AUC.
- Improve the user interface and visualizations.
- Compare different classification models.
- Add customer retention recommendations.
- Improve input validation and error handling.

## 👨‍💻 Author

**Harshad Sindhav**

GitHub: [harshad-sindhav](https://github.com/harshad-sindhav)

## 📄 License

See the `LICENSE` file in this repository for licensing information.
