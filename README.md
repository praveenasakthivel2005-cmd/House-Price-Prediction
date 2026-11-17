# 🏠 House Price Prediction Using Machine Learning

## 📌 Project Overview

House Price Prediction is a Machine Learning project that predicts house prices based on property details such as city, location, house area, and number of bedrooms.

This project uses a trained Machine Learning model and an interactive Streamlit web application to estimate house prices.

## 🎯 Project Objectives

- Predict house prices using Machine Learning.
- Analyze housing dataset information.
- Understand the relationship between house area and price.
- Provide an interactive web application.
- Generate estimated house prices based on user inputs.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- Matplotlib
- Seaborn

## 📂 Project Structure

```text
House-Price-Prediction/
│
├── app.py
├── Home_price_predictions.ipynb
├── csvdata.csv
├── best_home_price_model.pkt
├── requirements.txt
├── README.md
└── .gitignore
```

## 📊 Dataset Information

Dataset: `csvdata.csv`

Total Properties: 29,135

Features:
- City
- Location
- Area
- Bedrooms
- Price

Target Variable: Price

## ⚙️ Project Workflow

1. Import required Python libraries.
2. Load the housing dataset.
3. Perform data cleaning and preprocessing.
4. Explore the dataset.
5. Train Machine Learning regression models.
6. Evaluate model performance.
7. Save the trained model using Joblib.
8. Build a Streamlit web application.
9. Predict house prices using user inputs.

## 💻 Application Features

- Interactive Streamlit interface.
- City selection.
- Location selection.
- House area input.
- Number of bedrooms selection.
- House price prediction.
- Estimated price display.

## 🚀 Installation and Execution

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR-USERNAME/House-Price-Prediction.git
```

### Step 2: Open Project Folder

```bash
cd House-Price-Prediction
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Application

```bash
streamlit run app.py
```

## 📈 Expected Output

The application displays:

- Prediction status
- Estimated house price
- Selected city
- House area
- Number of bedrooms

## 🔍 Project Result

The trained Machine Learning model predicts estimated house prices based on the property details provided by the user.

## ⚠️ Disclaimer

This is an ML-based estimate from the supplied dataset, not an official property valuation or market quote.

## 👩‍💻 Author

**G. Praveena**

Machine Learning | Python | Data Science

## ⭐ Acknowledgement

This project was developed as a Machine Learning learning project to gain practical experience in data preprocessing, model training, prediction, and web application development.
