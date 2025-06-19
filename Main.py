import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Model
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

st.set_page_config(page_title="ML Model Tester", layout="wide")
st.title("🔍 Machine Learning Model Tester (Classifier & Regressor)")
st.title("Tugas Besar Dasar Ilmu Data Kelompok 4 ")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload dataset CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Tangani NaN
    if df.isnull().values.any():
        st.warning("⚠️ Dataset mengandung nilai kosong (NaN). Akan diisi otomatis:")
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean(numeric_only=True))

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Pilih target kolom
    columns = df.columns.tolist()
    target = st.selectbox("🎯 Pilih kolom target", columns)

    problem_type = st.radio("🧠 Jenis tugas:", ["Klasifikasi", "Regresi"])
    test_size = st.slider("Test Size (%)", 10, 50, 30)

    X = df.drop(columns=[target])
    y = df[target]

    # Encode data object
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # Normalisasi
    if st.checkbox("⚙️ Gunakan StandardScaler (Normalisasi fitur)"):
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42)

    # Pilih model
    model_name = st.selectbox("🔧 Pilih model", [
        "Decision Tree", "K-Nearest Neighbors", "Support Vector Machine", "Neural Network"
    ])

    def get_model(name, task):
        if name == "Decision Tree":
            return DecisionTreeClassifier() if task == "Klasifikasi" else DecisionTreeRegressor()
        elif name == "K-Nearest Neighbors":
            return KNeighborsClassifier() if task == "Klasifikasi" else KNeighborsRegressor()
        elif name == "Support Vector Machine":
            return SVC(probability=True) if task == "Klasifikasi" else SVR()
        elif name == "Neural Network":
            return MLPClassifier(max_iter=500) if task == "Klasifikasi" else MLPRegressor(max_iter=500)

    model = get_model(model_name, problem_type)

    if st.button("🚀 Jalankan Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if problem_type == "Klasifikasi":
            st.subheader("📈 Hasil Klasifikasi")
            st.text(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm).plot(ax=ax)
            st.pyplot(fig)

        else:
            st.subheader("📉 Hasil Regresi")
            st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, color="blue", alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)
