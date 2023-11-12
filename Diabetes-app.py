#from json import encoder
#from logging import LogRecord
#from quopri import encodestring
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV



#st.set_page_config(layout="wide")
st.title('Diabetes Prediction')

# Load and display data
@st.cache_data
def load_data():
    df = pd.read_csv('diabetes-data.csv')
    return df

data = load_data()

# Create tabs
tabs = ["Data Exploration", "Data Preprocessing and Model Training/Evaluation", "Prediction"]
selected_tab = st.sidebar.selectbox("Select Process", tabs)

# Data Exploration Process
if selected_tab == "Data Exploration":
    #st.header('Data Exploration Process')
    st.subheader('Diabetes Data')
    st.dataframe(data)

    # Data exploration steps
    st.subheader('Descriptive Statistics')
    st.write(data.describe())

    # ... (previous code)

elif selected_tab == "Data Preprocessing and Model Training/Evaluation":
    st.subheader('Data Preprocessing and Model Training/Evaluation')

    def preprocess_data(data):
        # Handle missing values if necessary
        # Example: Replace zeros in 'Glucose', 'BloodPressure', etc., with NaN
        # data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)

        imputer = SimpleImputer(strategy='mean')  # Adjust imputation strategy as needed
        data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        # Scale all numerical features
        scaler = StandardScaler()
        numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        data_imputed[numerical_features] = scaler.fit_transform(data_imputed[numerical_features])

        X = data_imputed.drop('Outcome', axis=1)  # Replace with your target column name
        y = data_imputed['Outcome']  # Replace with your target column name

        return X, y

    X, y = preprocess_data(data)

    if X.isnull().sum().sum() > 0:
        st.write('There are still NaN values in the data. Please check the preprocessing steps.')
    else:
        st.subheader('Preprocessed Data')
        st.dataframe(X.head())

        # Add code for model training and evaluation here

# ...
#elif selected_tab == "Model Training/Evaluation":

# Train model
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        logreg = LogisticRegression()
        svm = SVC()
        gbm = GradientBoostingClassifier()

        model.fit(X_train, y_train)
        logreg.fit(X_train, y_train)
        svm.fit(X_train, y_train)
        gbm.fit(X_train, y_train)

        # Perform undersampling on the training set
        undersampler = RandomUnderSampler()
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

        # Train models on the resampled training set
        model.fit(X_train_resampled, y_train_resampled)
        logreg.fit(X_train_resampled, y_train_resampled)
        svm.fit(X_train_resampled, y_train_resampled)
        gbm.fit(X_train_resampled, y_train_resampled)

        # Calculate training accuracy and validation accuracy for each model
        train_accuracy_model = accuracy_score(y_train_resampled, model.predict(X_train_resampled))
        val_accuracy_model = accuracy_score(y_val, model.predict(X_val))

        train_accuracy_logreg = accuracy_score(y_train_resampled, logreg.predict(X_train_resampled))
        val_accuracy_logreg = accuracy_score(y_val, logreg.predict(X_val))

        train_accuracy_svm = accuracy_score(y_train_resampled, svm.predict(X_train_resampled))
        val_accuracy_svm = accuracy_score(y_val, svm.predict(X_val))

        train_accuracy_gbm = accuracy_score(y_train_resampled, gbm.predict(X_train_resampled))
        val_accuracy_gbm = accuracy_score(y_val, gbm.predict(X_val))

         # Evaluate performance on validation set
        val_predictions_model = model.predict(X_val)
        val_accuracy_model = accuracy_score(y_val, val_predictions_model)

        val_predictions_logreg = logreg.predict(X_val)
        val_accuracy_logreg = accuracy_score(y_val, val_predictions_logreg)

        val_predictions_svm = svm.predict(X_val)
        val_accuracy_svm = accuracy_score(y_val, val_predictions_svm)

        val_predictions_gbm = gbm.predict(X_val)
        val_accuracy_gbm = accuracy_score(y_val, val_predictions_gbm)

       # Display training accuracy and validation accuracy for each model
        st.write(f'Training Accuracy (Random Forest): {train_accuracy_model}')
        st.write(f'Validation Accuracy (Random Forest): {val_accuracy_model}')
        st.write(f'Training Accuracy (Logistic Regression): {train_accuracy_logreg}')
        st.write(f'Validation Accuracy (Logistic Regression): {val_accuracy_logreg}')
        st.write(f'Training Accuracy (SVM): {train_accuracy_svm}')
        st.write(f'Validation Accuracy (SVM): {val_accuracy_svm}')
        st.write(f'Training Accuracy (Gradient Boosting): {train_accuracy_gbm}')
        st.write(f'Validation Accuracy (Gradient Boosting): {val_accuracy_gbm}')


 
        # Learning curve for Random Forest
        train_sizes_rf, train_scores_rf, val_scores_rf = learning_curve(model, X, y, cv=5)
        train_mean_rf = np.mean(train_scores_rf, axis=1)
        train_std_rf = np.std(train_scores_rf, axis=1)
        val_mean_rf = np.mean(val_scores_rf, axis=1)
        val_std_rf = np.std(val_scores_rf, axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes_rf, train_mean_rf, label='Training Accuracy (Random Forest)')
        plt.fill_between(train_sizes_rf, train_mean_rf - train_std_rf, train_mean_rf + train_std_rf, alpha=0.3)
        plt.plot(train_sizes_rf, val_mean_rf, label='Validation Accuracy (Random Forest)')
        plt.fill_between(train_sizes_rf, val_mean_rf - val_std_rf, val_mean_rf + val_std_rf, alpha=0.3)

        # Learning curve for Logistic Regression
        train_sizes_lr, train_scores_lr, val_scores_lr = learning_curve(logreg, X, y, cv=5)
        train_mean_lr = np.mean(train_scores_lr, axis=1)
        train_std_lr = np.std(train_scores_lr, axis=1)
        val_mean_lr = np.mean(val_scores_lr, axis=1)
        val_std_lr = np.std(val_scores_lr, axis=1)

        plt.plot(train_sizes_lr, train_mean_lr, label='Training Accuracy (Logistic Regression)')
        plt.fill_between(train_sizes_lr, train_mean_lr - train_std_lr, train_mean_lr + train_std_lr, alpha=0.3)
        plt.plot(train_sizes_lr, val_mean_lr, label='Validation Accuracy (Logistic Regression)')
        plt.fill_between(train_sizes_lr, val_mean_lr - val_std_lr, val_mean_lr + val_std_lr, alpha=0.3)

        # Learning curve for SVM
        train_sizes_svm, train_scores_svm, val_scores_svm = learning_curve(svm, X, y, cv=5)
        train_mean_svm = np.mean(train_scores_svm, axis=1)
        train_std_svm = np.std(train_scores_svm, axis=1)
        val_mean_svm = np.mean(val_scores_svm, axis=1)
        val_std_svm = np.std(val_scores_svm, axis=1)

        plt.plot(train_sizes_svm, train_mean_svm, label='Training Accuracy (SVM)')
        plt.fill_between(train_sizes_svm, train_mean_svm - train_std_svm, train_mean_svm + train_std_svm, alpha=0.3)
        plt.plot(train_sizes_svm, val_mean_svm, label='Validation Accuracy (SVM)')
        plt.fill_between(train_sizes_svm, val_mean_svm - val_std_svm, val_mean_svm + val_std_svm, alpha=0.3)

       # Learning curve for Gradient Boosting
        train_sizes_gbm, train_scores_gbm, val_scores_gbm = learning_curve(gbm, X, y, cv=5)
        train_mean_gbm = np.mean(train_scores_gbm, axis=1)
        train_std_gbm = np.std(train_scores_gbm, axis=1)
        val_mean_gbm = np.mean(val_scores_gbm, axis=1)
        val_std_gbm = np.std(val_scores_gbm, axis=1)

        plt.plot(train_sizes_gbm, train_mean_gbm, label='Training Accuracy (Gradient Boosting)')
        plt.fill_between(train_sizes_gbm, train_mean_gbm - train_std_gbm, train_mean_gbm + train_std_gbm, alpha=0.3)
        plt.plot(train_sizes_gbm, val_mean_gbm, label='Validation Accuracy (Gradient Boosting)')
        plt.fill_between(train_sizes_gbm, val_mean_gbm - val_std_gbm, val_mean_gbm + val_std_gbm, alpha=0.3)

        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title('Learning Curve')
        plt.legend()
        st.pyplot(plt)


       # Ensure y_val and predictions are of integer type
        y_val_int = y_val.astype(int)
        predictions_rf_int = val_predictions_model.astype(int)
        predictions_lr_int = val_predictions_logreg.astype(int)
        predictions_svm_int = val_predictions_svm.astype(int)
        predictions_gbm_int = val_predictions_gbm.astype(int)

# Calculate evaluation metrics and confusion matrix for Random Forest
        accuracy_rf = accuracy_score(y_val_int, predictions_rf_int)
        precision_rf = precision_score(y_val_int, predictions_rf_int, pos_label=1)
        recall_rf = recall_score(y_val_int, predictions_rf_int, pos_label=1)
        f1_rf = f1_score(y_val_int, predictions_rf_int, pos_label=1)
        cm_rf = confusion_matrix(y_val_int, predictions_rf_int)

# ... (Repeat the same for Logistic Regression, SVM, and Gradient Boosting)

# Logistic Regression
    accuracy_lr = accuracy_score(y_val_int, predictions_lr_int)
    precision_lr = precision_score(y_val_int, predictions_lr_int, pos_label=1)
    recall_lr = recall_score(y_val_int, predictions_lr_int, pos_label=1)
    f1_lr = f1_score(y_val_int, predictions_lr_int, pos_label=1)
    cm_lr = confusion_matrix(y_val_int, predictions_lr_int)

# SVM
    accuracy_svm = accuracy_score(y_val_int, predictions_svm_int)
    precision_svm = precision_score(y_val_int, predictions_svm_int, pos_label=1)
    recall_svm = recall_score(y_val_int, predictions_svm_int, pos_label=1)
    f1_svm = f1_score(y_val_int, predictions_svm_int, pos_label=1)
    cm_svm = confusion_matrix(y_val_int, predictions_svm_int)

# Gradient Boosting
    accuracy_gbm = accuracy_score(y_val_int, predictions_gbm_int)
    precision_gbm = precision_score(y_val_int, predictions_gbm_int, pos_label=1)
    recall_gbm = recall_score(y_val_int, predictions_gbm_int, pos_label=1)
    f1_gbm = f1_score(y_val_int, predictions_gbm_int, pos_label=1)
    cm_gbm = confusion_matrix(y_val_int, predictions_gbm_int)



    # Display evaluation metrics and confusion matrix for Random Forest
    st.subheader('Model Evaluation Metrics (Random Forest)')
    st.write(f'Accuracy: {accuracy_rf}')
    st.write(f'Precision: {precision_rf}')
    st.write(f'Recall: {recall_rf}')
    st.write(f'F1-score: {f1_rf}')

    st.subheader('Confusion Matrix (Random Forest)')
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_rf, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

     # Display evaluation metrics and confusion matrix for Logistic Regression
    st.subheader('Model Evaluation Metrics (Logistic Regression)')
    st.write(f'Accuracy: {accuracy_lr}')
    st.write(f'Precision: {precision_lr}')
    st.write(f'Recall: {recall_lr}')
    st.write(f'F1-score: {f1_lr}')

    st.subheader('Confusion Matrix (Logistic Regression)')
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_lr, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

        # Display evaluation metrics and confusion matrix for SVM
    st.subheader('Model Evaluation Metrics (SVM)')
    st.write(f'Accuracy: {accuracy_svm}')
    st.write(f'Precision: {precision_svm}')
    st.write(f'Recall: {recall_svm}')
    st.write(f'F1-score: {f1_svm}')

    st.subheader('Confusion Matrix (SVM)')
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_svm, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

        # Display evaluation metrics and confusion matrix for Gradient Boosting
    st.subheader('Model Evaluation Metrics (Gradient Boosting)')
    st.write(f'Accuracy: {accuracy_gbm}')
    st.write(f'Precision: {precision_gbm}')
    st.write(f'Recall: {recall_gbm}')
    st.write(f'F1-score: {f1_gbm}')

    st.subheader('Confusion Matrix (Gradient Boosting)')
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_gbm, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

      
 # # ... (previous code remains the same)


elif selected_tab == "Prediction":
    st.subheader('Make a Prediction')

    # Create input fields for each feature
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=117)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=122, value=72)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=99, value=20)
    insulin = st.number_input('Insulin', min_value=0, max_value=846, value=79)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=32.0)
    diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input('Age', min_value=21, max_value=100, value=33)

    # Button to make prediction
    if st.button('Predict Diabetes'):
        # Combine inputs into a single dataframe
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        input_df = pd.DataFrame(input_data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

        # Process user input here similarly to how training data was processed
        # Example:
        # processed_input = scaler.transform(input_df) # assuming 'scaler' is a pre-fitted StandardScaler object

        # Check if preprocessing step is defined
        if 'processed_input' in locals():
            # Use the trained model for prediction
            prediction = logreg.predict(processed_input)  # Example using RandomForest
            st.subheader('Prediction:')
            st.write('Diabetes' if prediction[0] == 1 else 'No Diabetes')
        else:
            st.error("Processed input is not defined. Please check your preprocessing steps.")



