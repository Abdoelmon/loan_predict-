# Loan Prediction using Machine Learning

## 📌 Project Overview
This project aims to predict **whether a loan will be approved or not** based on applicant details such as:
- Gender  
- Marital status  
- Number of dependents  
- Education level  
- Self-employed status  
- Applicant income  
- Coapplicant income  
- Loan amount and loan term  
- Credit history  
- Property area  

Several machine learning models were trained and compared, and the best-performing model was deployed as an interactive **Streamlit web application**.

---

## 📂 Project Structure

---
├── loan-prediction-w-various-ml-models.ipynb # Jupyter Notebook: training and experiments
├── deploy.py # Streamlit app for deployment
├── model.pkl # Trained model (saved with joblib)
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Requirements
Install the required libraries:

```bash
pip install -r requirements.txt
Example requirements.txt
nginx
Copy code
streamlit
pandas
numpy
scikit-learn
joblib
🚀 How to Run
1. Train the Model (Optional)
Open the Jupyter Notebook and run the training code:

bash
Copy code
jupyter notebook loan-prediction-w-various-ml-models.ipynb
After training, save the model as model.pkl.

2. Run the Streamlit App
bash
Copy code
streamlit run deploy.py
3. Access the Application
Open the link shown in the terminal (usually: http://localhost:8501) to use the app.

🖼️ Application Features
User-friendly interface with dropdowns and input fields for applicant details.

Click Predict to get the result.

The app will display:

✅ Loan Approved → if the loan application is predicted to be approved.

❌ Loan Not Approved → if the loan is predicted to be rejected.

📊 Machine Learning Models
The following algorithms were tested during experimentation:

Logistic Regression

Decision Trees

Random Forest

Support Vector Machines (SVM)

XGBoost

The final model was selected based on performance metrics such as Accuracy, Precision, and Recall.

🌐 Deployment
This app can be deployed on platforms such as:

Streamlit Cloud
