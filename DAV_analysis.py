import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from scipy.stats import chi2_contingency

st.sidebar.header("Select Analysis")
options = [
    "Data Overview", 
    "Distribution Analysis", 
    "Correlation Analysis", 
    "Predictive Modeling", 
    "Heart Attack Risk by Diabetes Status", 
    "Subgroup Analysis by Diabetes Status"
]
choice = st.sidebar.radio("Go to", options)

file_path = "C:\\Users\\aabha\\Downloads\\heart_attack_prediction_dataset.csv"  
data = pd.read_csv(file_path)

if choice == "Data Overview":
    st.header("Data Overview")
    st.write("Preview of the dataset:")
    st.write(data.head())
    st.write("Summary Statistics:")
    st.write(data.describe())

elif choice == "Distribution Analysis":
    st.header("Distribution Analysis")
    distribution_columns = ['Age', 'Heart Rate', 'BMI', 'Cholesterol', 'Sleep Hours Per Day']
    st.write("Distribution of Selected Continuous Features:")
    fig, axs = plt.subplots(len(distribution_columns), 1, figsize=(10, len(distribution_columns) * 2))
    for i, col in enumerate(distribution_columns):
        sns.histplot(data[col].dropna(), kde=True, ax=axs[i])
        axs[i].set_title(f"Distribution of {col}")
    plt.tight_layout()
    st.pyplot(fig)

elif choice == "Correlation Analysis":
    st.header("Correlation Analysis")
    numerical_data = data.select_dtypes(include=['float64', 'int64']).drop(
        columns=[col for col in ['Patient ID', 'Sex', 'Country', 'Continent', 'Hemisphere', 'Diet'] if col in data.columns],
        errors='ignore'  
    )
    correlation_data = numerical_data.corr().round(2)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_data, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Correlation Heatmap")
    st.write("Correlation Heatmap of all numerical features:")
    st.pyplot(fig)

elif choice == "Predictive Modeling":
    st.header("Heart Attack Prediction Model")

    data = data.drop(columns=['Patient ID', 'Country', 'Continent', 'Hemisphere', 'Blood Pressure'], errors='ignore')

    label_encoder = LabelEncoder()
    for col in ['Sex', 'Diet']:  
        if col in data.columns:
            data[col] = label_encoder.fit_transform(data[col])

    X = data.drop(columns=['Heart Attack Risk'])
    y = data['Heart Attack Risk']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=15)  
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.write(f"Model Accuracy on Test Set: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(report)

elif choice == "Heart Attack Risk by Diabetes Status":
    st.header("Heart Attack Risk by Diabetes Status")

    diabetes_heart_risk_table = pd.crosstab(data['Diabetes'], data['Heart Attack Risk'])
    st.write("Contingency Table:")
    st.write(diabetes_heart_risk_table)

    chi2, p, _, _ = chi2_contingency(diabetes_heart_risk_table)
    st.write(f"Chi-square Test p-value: {p:.4f}")
    if p < 0.05:
        st.write("There is a significant association between Diabetes and Heart Attack Risk.")
    else:
        st.write("There is no significant association between Diabetes and Heart Attack Risk.")

    fig, ax = plt.subplots(figsize=(8, 6))
    diabetes_heart_risk_table.plot(kind="bar", stacked=True, ax=ax)
    plt.title("Heart Attack Risk by Diabetes Status")
    plt.xlabel("Diabetes")
    plt.ylabel("Count")
    st.pyplot(fig)

elif choice == "Subgroup Analysis by Diabetes Status":
    st.header("Subgroup Analysis by Diabetes Status")

    subgroup_columns = ['Cholesterol', 'Heart Rate', 'BMI']

    fig, axs = plt.subplots(len(subgroup_columns), 1, figsize=(10, len(subgroup_columns) * 4))
    for i, col in enumerate(subgroup_columns):
        sns.boxplot(x='Diabetes', y=col, data=data, ax=axs[i])
        axs[i].set_title(f"{col} Levels by Diabetes Status")
    plt.tight_layout()
    st.pyplot(fig)
