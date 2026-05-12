import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
import joblib

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Borana University AI System",
    page_icon="🎓",
    layout="wide"
)

# ==========================================================
# KEEP YOUR ORIGINAL COLORS
# ==========================================================
st.markdown("""
<style>

.stApp {
    background-color: #f4f7f6;
}

/* HEADER */
.main-header {
    background-color: #004d99;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}

/* KPI CARDS */
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border-left: 8px solid #004d99;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.1);
}

/* SECTION BOX */
.section-box {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
}

/* RESULT BOX */
.result-box {
    background-color: white;
    border-top: 10px solid #004d99;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}

/* BUTTON */
.stButton>button {
    width: 100%;
    height: 3.2em;
    border-radius: 10px;
    background-color: #004d99;
    color: white;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LABELS
# ==========================================================
ETHNICITY_LABELS = {
    0: 'Oromia',
    1: 'Amhara',
    2: 'Somalia',
    3: 'Tigray',
    4: 'Sidama',
    5: 'Afar',
    6: 'other'
}

EDUCATION_LABELS = {
    0: 'None',
    1: 'High School',
    2: 'Some College',
    3: 'Bachelors',
    4: 'Higher'
}

SUPPORT_LABELS = {
    0: 'None',
    1: 'Low',
    2: 'Medium',
    3: 'High',
    4: 'Very High'
}

GRADE_LABELS = {
    0: 'A (Excellent)',
    1: 'B (Good)',
    2: 'C (Average)',
    3: 'D (Pass)',
    4: 'F (Fail)'
}

# ==========================================================
# SYSTEM CLASS
# ==========================================================
class BoranaSystem:

    def __init__(self):

        self.data_file = "Student_performance_data-_.csv"

        self.model_file = "borana_ai_model.joblib"

        self.features = [
            'Age',
            'Gender',
            'Ethnicity',
            'ParentalEducation',
            'StudyTimeWeekly',
            'Absences',
            'Tutoring',
            'ParentalSupport',
            'Extracurricular',
            'Sports',
            'Music',
            'Volunteering'
        ]

    @st.cache_data
    def load_data(_self):

        if os.path.exists(_self.data_file):

            df = pd.read_csv(_self.data_file)

            if "StudentID" in df.columns:
                df = df.drop("StudentID", axis=1)

            return df

        return None

    def train_model(self, df, n_trees):

        X = df[self.features]

        y = df['GradeClass']

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=n_trees,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        report = classification_report(
            y_test,
            predictions,
            output_dict=True
        )

        cm = confusion_matrix(y_test, predictions)

        joblib.dump(model, self.model_file)

        return accuracy, report, cm

    def predict(self, input_df):

        if os.path.exists(self.model_file):

            model = joblib.load(self.model_file)

            prediction = model.predict(input_df)[0]

            return prediction

        return None

# ==========================================================
# HEADER
# ==========================================================
def render_header():
    # Logo fiduu fi gidduu galchuun asirratti dabalameera
    if os.path.exists("OIP.webp"):
        logo = Image.open("OIP.webp")
        left_co, cent_co, last_co = st.columns([1, 1, 1])
        with cent_co:
            st.image(logo, width=150)

    st.markdown("""
    <div class="main-header">
        <h1>🎓 BORANA UNIVERSITY</h1>
        <p>AI Based Student Performance Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# MAIN FUNCTION
# ==========================================================
def main():

    render_header()

    system = BoranaSystem()

    df = system.load_data()

    if df is None:

        st.error("Dataset not found!")

        st.info("Place CSV file inside project folder.")

        return

    # ======================================================
    # SIDEBAR
    # ======================================================
    st.sidebar.title("📌 Navigation")

    menu = st.sidebar.radio(
        "Select Module",
        [
            "🏠 Dashboard",
            "📊 Analytics Center",
            "⚙️ AI Training Lab",
            "🔮 Prediction Center",
            "ℹ️ System Information"
        ]
    )

    st.sidebar.markdown("---")

    st.sidebar.success("Borana University")

    st.sidebar.info("Department of Computer Science")

    # ======================================================
    # DASHBOARD
    # ======================================================
    if menu == "🏠 Dashboard":

        st.subheader("📊 System Dashboard")

        # KPI CARDS
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{len(df)}</h2>
                <p>Total Students</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{len(df.columns)}</h2>
                <p>Attributes</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
                <h2>RF</h2>
                <p>AI Algorithm</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="metric-card">
                <h2>ML</h2>
                <p>Machine Learning</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # QUICK ANALYTICS
        st.subheader("📈 Quick Analytics")

        c1, c2 = st.columns(2)

        with c1:

            fig, ax = plt.subplots(figsize=(6,4))

            sns.countplot(
                x='GradeClass',
                data=df,
                ax=ax
            )

            ax.set_title("Grade Distribution")

            st.pyplot(fig)

        with c2:

            fig, ax = plt.subplots(figsize=(6,4))

            sns.boxplot(
                x='GradeClass',
                y='StudyTimeWeekly',
                data=df,
                ax=ax
            )

            ax.set_title("Study Time vs Grade")

            st.pyplot(fig)

        # DATA PREVIEW
        with st.container():

            st.subheader("📋 Recent Student Records")

            st.dataframe(
                df.head(20),
                width='stretch'
            )

    # ======================================================
    # ANALYTICS CENTER
    # ======================================================
    elif menu == "📊 Analytics Center":

        st.subheader("📊 Advanced Data Analytics")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Grade Distribution",
            "Correlation Matrix",
            "Feature Analysis",
            "Dataset Statistics"
        ])

        # --------------------------------------------------
        # TAB 1
        # --------------------------------------------------
        with tab1:

            st.markdown("### Grade Distribution Analysis")

            fig, ax = plt.subplots(figsize=(12,5))

            sns.countplot(
                x='GradeClass',
                data=df,
                ax=ax
            )

            ax.set_xticks(range(5))
            ax.set_xticklabels([
                GRADE_LABELS[i] for i in range(5)
            ])

            st.pyplot(fig)

        # --------------------------------------------------
        # TAB 2
        # --------------------------------------------------
        with tab2:

            st.markdown("### Feature Correlation Matrix")

            fig, ax = plt.subplots(figsize=(14,10))

            sns.heatmap(
                df.corr(),
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                ax=ax
            )

            st.pyplot(fig)

        # --------------------------------------------------
        # TAB 3
        # --------------------------------------------------
        with tab3:

            st.markdown("### Feature Impact Analysis")

            feature = st.selectbox(
                "Select Feature",
                [
                    'Absences',
                    'StudyTimeWeekly',
                    'Age'
                ]
            )

            fig, ax = plt.subplots(figsize=(10,5))

            sns.boxplot(
                x='GradeClass',
                y=feature,
                data=df,
                ax=ax
            )

            st.pyplot(fig)

        # --------------------------------------------------
        # TAB 4
        # --------------------------------------------------
        with tab4:

            st.markdown("### Dataset Statistical Summary")

            st.dataframe(
                df.describe(),
                width='stretch'
            )

    # ======================================================
    # AI TRAINING LAB
    # ======================================================
    elif menu == "⚙️ AI Training Lab":

        st.subheader("⚙️ Artificial Intelligence Training Center")

        with st.container():

            st.markdown("""
            <div class="section-box">

            Configure and train the AI model using Random Forest Algorithm.

            </div>
            """, unsafe_allow_html=True)

        trees = st.slider(
            "Select Number of Trees",
            50,
            500,
            150
        )

        if st.button("🚀 Train AI Model"):

            progress = st.progress(0)

            for i in range(100):

                time.sleep(0.01)

                progress.progress(i + 1)

            accuracy, report, cm = system.train_model(df, trees)

            st.success(
                f"Training Completed Successfully! Accuracy = {accuracy:.2%}"
            )

            # ACCURACY CARDS
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Accuracy",
                    f"{accuracy:.2%}"
                )

            with c2:
                st.metric(
                    "Algorithm",
                    "Random Forest"
                )

            with c3:
                st.metric(
                    "Trees",
                    trees
                )

            # CONFUSION MATRIX
            st.subheader("📉 Confusion Matrix")

            fig, ax = plt.subplots(figsize=(8,6))

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax
            )

            st.pyplot(fig)

            # REPORT
            with st.expander("📄 View Full Classification Report"):

                st.json(report)

    # ======================================================
    # PREDICTION CENTER
    # ======================================================
    elif menu == "🔮 Prediction Center":

        st.subheader("🔮 Student Performance Prediction")

        if not os.path.exists(system.model_file):

            st.warning("Please train the model first!")

            return

        tab1, tab2 = st.tabs([
            "📝 Student Form",
            "📘 Prediction Guide"
        ])

        # --------------------------------------------------
        # TAB 1
        # --------------------------------------------------
        with tab1:

            with st.form("prediction_form"):

                st.markdown("## 👤 Personal Information")

                col1, col2 = st.columns(2)

                with col1:

                    age = st.slider(
                        "Age",
                        15,
                        25,
                        18
                    )

                    gender = st.selectbox(
                        "Gender",
                        [0,1],
                        format_func=lambda x:
                        "Male" if x == 1 else "Female"
                    )

                    ethnicity = st.selectbox(
                        "Ethnicity",
                        [0,1,2,3],
                        format_func=lambda x:
                        ETHNICITY_LABELS[x]
                    )

                with col2:

                    parent_edu = st.selectbox(
                        "Parental Education",
                        [0,1,2,3,4],
                        format_func=lambda x:
                        EDUCATION_LABELS[x]
                    )

                    support = st.select_slider(
                        "Parental Support",
                        options=[0,1,2,3,4],
                        format_func=lambda x:
                        SUPPORT_LABELS[x]
                    )

                    tutor = st.radio(
                        "External Tutoring",
                        [0,1],
                        format_func=lambda x:
                        "Yes" if x == 1 else "No"
                    )

                st.markdown("---")

                st.markdown("## 📚 Academic Information")

                c1, c2 = st.columns(2)

                with c1:

                    study = st.slider(
                        "Weekly Study Time",
                        0,
                        40,
                        10
                    )

                with c2:

                    absent = st.slider(
                        "Absences",
                        0,
                        50,
                        5
                    )

                st.markdown("---")

                st.markdown("## 🎯 Activities")

                cc1, cc2, cc3, cc4 = st.columns(4)

                with cc1:
                    extra = st.checkbox("Extracurricular")

                with cc2:
                    sports = st.checkbox("Sports")

                with cc3:
                    music = st.checkbox("Music")

                with cc4:
                    volunteer = st.checkbox("Volunteering")

                st.markdown("<br>", unsafe_allow_html=True)

                submit = st.form_submit_button(
                    "🎯 Generate Prediction"
                )

            # PREDICTION
            if submit:

                input_df = pd.DataFrame([[
                    age,
                    gender,
                    ethnicity,
                    parent_edu,
                    study,
                    absent,
                    int(tutor),
                    support,
                    int(extra),
                    int(sports),
                    int(music),
                    int(volunteer)
                ]], columns=system.features)

                prediction = system.predict(input_df)

                if prediction is not None:

                    grade = GRADE_LABELS[int(prediction)]

                    st.markdown(f"""
                    <div class="result-box">

                    <h2>Prediction Result</h2>

                    <p>The student is predicted as:</p>

                    <h1 style="color:#004d99;">
                    {grade}
                    </h1>

                    </div>
                    """, unsafe_allow_html=True)

                    # EXTRA RESULT CARDS
                    rc1, rc2, rc3 = st.columns(3)

                    with rc1:
                        st.metric(
                            "Predicted Grade",
                            grade
                        )

                    with rc2:
                        st.metric(
                            "Model",
                            "Random Forest"
                        )

                    with rc3:
                        st.metric(
                            "Prediction Status",
                            "Completed"
                        )

                    if int(prediction) <= 1:
                        st.balloons()

        # --------------------------------------------------
        # TAB 2
        # --------------------------------------------------
        with tab2:

            st.info("""
            This AI system predicts student academic performance
            using machine learning techniques.

            Factors used include:

            - Study Time
            - Attendance
            - Tutoring
            - Activities
            - Parental Support
            - Education Background
            """)

    # ======================================================
    # SYSTEM INFORMATION
    # ======================================================
    elif menu == "ℹ️ System Information":

        st.subheader("ℹ️ System Details")

        with st.container():

            st.markdown("""
            ### 🎓 BORANA UNIVERSITY

            #### AI Student Performance Prediction System

            This project is developed using:

            - Streamlit
            - Python
            - Machine Learning
            - Random Forest Classifier
            - Pandas
            - Matplotlib
            - Seaborn

            #### Developed For:
            Department of Computer Science

            #### Location:
            Yabelo, Ethiopia
            """)

# ==========================================================
# RUN APPLICATION
# ==========================================================
if __name__ == "__main__":
    main()