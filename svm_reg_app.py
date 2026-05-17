import streamlit as st
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("student_placement_salary_(SVM).csv")

# Drop ID column
df.drop("student_id", axis=1, inplace=True)

# Encode categorical columns
le_branch = LabelEncoder()
le_company = LabelEncoder()
le_role = LabelEncoder()

df['branch'] = le_branch.fit_transform(df['branch'])
df['company_type'] = le_company.fit_transform(df['company_type'])
df['job_role'] = le_role.fit_transform(df['job_role'])

# Features and target
X = df.drop("salary_lpa", axis=1)
y = df["salary_lpa"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
score = r2_score(y_test, pred)

# ---------------- UI ----------------

st.title("Student Placement Salary Prediction")
st.subheader(f"Model Accuracy (R²): {score:.2f}")

cgpa = st.slider("CGPA", 5.0, 10.0, 7.5)
branch = st.selectbox("Branch", list(le_branch.classes_))
college_tier = st.selectbox("College Tier", [1,2,3])

python_skill = st.slider("Python Skill", 0,100,70)
dsa_skill = st.slider("DSA Skill", 0,100,70)
ml_skill = st.slider("ML Skill", 0,100,70)
web_dev_skill = st.slider("Web Dev Skill", 0,100,70)

coding_score = st.slider("Coding Score",0,100,75)
communication_score = st.slider("Communication Score",0,100,75)
aptitude_score = st.slider("Aptitude Score",0,100,75)

internships = st.slider("Internships",0,10,2)
projects = st.slider("Projects",0,20,5)
backlogs = st.slider("Backlogs",0,10,0)

resume_score = st.slider("Resume Score",0,100,75)
skill_score = st.slider("Skill Score",0,100,75)

placed = st.selectbox("Placed",[0,1])

company_type = st.selectbox(
    "Company Type",
    list(le_company.classes_)
)

job_role = st.selectbox(
    "Job Role",
    list(le_role.classes_)
)

# Input dataframe
input_data = pd.DataFrame([[
    cgpa,
    le_branch.transform([branch])[0],
    college_tier,
    python_skill,
    dsa_skill,
    ml_skill,
    web_dev_skill,
    coding_score,
    communication_score,
    aptitude_score,
    internships,
    projects,
    backlogs,
    resume_score,
    skill_score,
    placed,
    le_company.transform([company_type])[0],
    le_role.transform([job_role])[0]
]], columns=X.columns)

# Scale input
scaled_input = scaler.transform(input_data)

if st.button("Predict Salary"):

    prediction = model.predict(scaled_input)[0]

    # backlog penalty
    if backlogs >= 1:
        prediction *= 0.85

    if backlogs >= 2:
        prediction *= 0.75

    if backlogs >= 3:
        prediction *= 0.60

    if backlogs >= 5:
        prediction *= 0.40

    st.success(f"Predicted Salary Package: {prediction:.2f} LPA")