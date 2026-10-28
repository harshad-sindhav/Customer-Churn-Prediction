# Customer Churn Prediction using Deep Learning

## 📌 Project Overview

Customer Churn Prediction is a Machine Learning and Deep Learning project that predicts whether a bank customer is likely to leave the bank.

The project uses a trained neural network built with TensorFlow/Keras and provides an interactive web application using Streamlit. Users can enter customer details and view the model's predicted churn probability.

## 🎯 Objectives

- Predict customer churn using a trained Deep Learning model.
- Preprocess numerical and categorical customer features.
- Apply feature scaling and categorical encoding.
- Build an interactive prediction interface using Streamlit.
- Demonstrate an end-to-end Machine Learning workflow.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Jupyter Notebook

## 📂 Project Structure

```text
Customer-Churn-Prediction/
├── Churn_Modelling.csv
├── LICENSE
├── README.md
├── algo.ipynb
├── app.py
├── label_gender.pkl
├── model.h5
├── one_hot_geo.pkl
├── prediction1.ipynb
├── requirements.txt
└── scaler.pkl
```

## 📊 Dataset

The project uses `Churn_Modelling.csv`, which contains bank customer information and the target column `Exited`.

Features include:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Membership Status
- Estimated Salary

The target variable indicates whether a customer left the bank.

## ⚙️ Project Workflow

1. Collect customer details through the application.
2. Encode the Gender feature using the saved label encoder.
3. Encode Geography using the saved One-Hot Encoder.
4. Scale input features using the saved scaler.
5. Pass the processed features to the trained neural network.
6. Display the predicted churn probability and classification.

## 📁 Important Files

| File | Description |
|---|---|
| `app.py` | Streamlit web application |
| `model.h5` | Trained TensorFlow/Keras model |
| `scaler.pkl` | Saved feature scaler |
| `label_gender.pkl` | Saved gender encoder |
| `one_hot_geo.pkl` | Saved geography encoder |
| `Churn_Modelling.csv` | Dataset |
| `algo.ipynb` | Model development notebook |
| `prediction1.ipynb` | Prediction and preprocessing notebook |
| `requirements.txt` | Project dependencies |

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/harshad-sindhav/Customer-Churn-Prediction.git
```

### 2. Open the project directory

```bash
cd Customer-Churn-Prediction
```

### 3. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it in Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal if the application does not open automatically.

## 🚀 Streamlit Cloud Deployment

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in using your GitHub account.
3. Create a new app.
4. Select the repository `harshad-sindhav/Customer-Churn-Prediction`.
5. Choose the `main` branch.
6. Set the main file path to `app.py`.
7. Click **Deploy**.

Ensure that `requirements.txt` contains compatible dependencies and that all model and preprocessing files are available in the repository.

## 📈 Prediction Output

The application displays a churn probability and classifies the customer based on the threshold configured in the code.

- **Likely to Leave:** The predicted probability meets or exceeds the configured threshold.
- **Likely to Stay:** The predicted probability is below the configured threshold.

Predictions are estimates and should not be treated as guaranteed outcomes.

## 🔮 Future Improvements

- Evaluate the model using accuracy, precision, recall, F1-score, and ROC-AUC.
- Improve the user interface and data visualizations.
- Compare different classification algorithms.
- Add customer retention recommendations.
- Improve input validation and error handling.

## 👨‍💻 Author

**Harshad Sindhav**

GitHub: [harshad-sindhav](https://github.com/harshad-sindhav)

## 📄 License




This project is distributed under the license specified in the `LICENSE` file.
