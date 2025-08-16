import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ----------------- Load Data -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("job_salary.csv")

    # Remove irrelevant currency column if exists
    if "salary_currency" in df.columns:
        df = df.drop(columns=["salary_currency"])

    # Handle outliers in salary
    lower, upper = df["salary_in_usd"].quantile([0.01, 0.99])
    df["salary_in_usd"] = df["salary_in_usd"].clip(lower, upper)

    return df

df_raw = load_data()


# ----------------- Mapping -----------------
experience_mapping = {
    "EX": "EX → Executive-level / Director",
    "SE": "SE → Senior-level / Expert",
    "MI": "MI → Mid-level / Intermediate",
    "EN": "EN → Entry-level / Junior"
}

employment_mapping = {
    "FT": "FT → Full-time",
    "PT": "PT → Part-time",
    "CT": "CT → Contract",
    "FL": "FL → Freelance"
}

company_size_mapping = {
    "S": "S → Small (<50 employees)",
    "M": "M → Medium (50-250 employees)",
    "L": "L → Large (>250 employees)"
}

country_mapping = {
    "US": "United States", "CA": "Canada", "NL": "Netherlands", "FR": "France", "GB": "United Kingdom",
    "DE": "Germany", "ES": "Spain", "IE": "Ireland", "AU": "Australia", "LT": "Lithuania",
    "IN": "India", "EE": "Estonia", "SK": "Slovakia", "CZ": "Czech Republic", "UA": "Ukraine",
    "FI": "Finland", "CO": "Colombia", "AR": "Argentina", "AT": "Austria", "EG": "Egypt",
    "SG": "Singapore", "MX": "Mexico", "IT": "Italy", "PL": "Poland", "BE": "Belgium",
    "CH": "Switzerland", "GR": "Greece", "NZ": "New Zealand", "HR": "Croatia", "PR": "Puerto Rico",
    "PT": "Portugal", "BR": "Brazil", "PH": "Philippines", "RO": "Romania", "ML": "Mali",
    "NO": "Norway", "CL": "Chile", "MY": "Malaysia", "SV": "El Salvador", "DO": "Dominican Republic",
    "GT": "Guatemala", "CR": "Costa Rica", "LV": "Latvia", "ZA": "South Africa", "JO": "Jordan",
    "CY": "Cyprus", "TH": "Thailand", "JM": "Jamaica", "JP": "Japan", "MT": "Malta",
    "MK": "North Macedonia", "SI": "Slovenia", "HK": "Hong Kong", "LS": "Lesotho", "ID": "Indonesia",
    "PE": "Peru", "HU": "Hungary", "PA": "Panama", "LU": "Luxembourg", "DZ": "Algeria",
    "KE": "Kenya", "CD": "Democratic Republic of the Congo", "SE": "Sweden", "KR": "South Korea",
    "TW": "Taiwan", "TR": "Turkey", "NG": "Nigeria", "DK": "Denmark", "AE": "United Arab Emirates",
    "BG": "Bulgaria", "RS": "Serbia", "EC": "Ecuador", "XK": "Kosovo", "ZM": "Zambia",
    "AM": "Armenia", "RW": "Rwanda", "IL": "Israel", "LB": "Lebanon", "PK": "Pakistan",
    "HN": "Honduras", "VE": "Venezuela", "BM": "Bermuda", "VN": "Vietnam", "GE": "Georgia",
    "SA": "Saudi Arabia", "OM": "Oman", "BA": "Bosnia and Herzegovina", "UG": "Uganda",
    "MU": "Mauritius", "QA": "Qatar", "RU": "Russia", "TN": "Tunisia", "GH": "Ghana",
    "AD": "Andorra", "MD": "Moldova", "UZ": "Uzbekistan", "CF": "Central African Republic",
    "KW": "Kuwait", "IR": "Iran", "AS": "American Samoa", "CN": "China", "BO": "Bolivia",
    "IQ": "Iraq", "JE": "Jersey"
}

# Apply mappings
df_raw["experience_level"] = df_raw["experience_level"].map(experience_mapping)
df_raw["employment_type"] = df_raw["employment_type"].map(employment_mapping)
df_raw["company_size"] = df_raw["company_size"].map(company_size_mapping)
df_raw["employee_residence"] = df_raw["employee_residence"].map(country_mapping)
df_raw["company_location"] = df_raw["company_location"].map(country_mapping)


# ----------------- Features & Target -----------------
y = np.log1p(df_raw["salary_in_usd"])   # log transform target
X = df_raw.drop("salary_in_usd", axis=1)

# Identify categorical & numeric columns
categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(exclude="object").columns

# Remove salary from numeric columns
numeric_cols = numeric_cols.drop("salary")

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

# ----------------- Models -----------------
models = {
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
    )
}

@st.cache_resource
def train_models():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trained_models = {}
    performance = []

    for name, model in models.items():
        pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        trained_models[name] = pipe

        # Predictions
        y_pred = np.expm1(pipe.predict(X_test))  # back-transform
        y_true = np.expm1(y_test)

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        performance.append({"Model": name, "MAE": mae, "R²": r2})

    return trained_models, pd.DataFrame(performance)

trained_models, perf_df = train_models()

# ----------------- UI -----------------
st.title("Job Salary Prediction App")
st.write("Predict salaries based on job role, location, and experience level.")

# Sidebar input
st.sidebar.header("Input Job Details")

def get_user_input():
    input_dict = {}
    for col in categorical_cols:
        input_dict[col] = st.sidebar.selectbox(col, df_raw[col].unique())
    for col in numeric_cols:
        if col == "remote_ratio":
            input_dict[col] = st.sidebar.slider("Remote Ratio (%)", 0, 100, 0)
        else:
            input_dict[col] = st.sidebar.number_input(
                col, float(df_raw[col].min()), float(df_raw[col].max())
            )
    return pd.DataFrame([input_dict])

user_df = get_user_input()

# Prediction
if st.sidebar.button("Predict Salary"):
    st.sidebar.subheader("Predicted Salary")
    preds = {}
    for name, model in trained_models.items():
        pred = np.expm1(model.predict(user_df))[0]
        preds[name] = pred
        st.sidebar.write(f"**{name}:** ${pred:,.2f}")

    # Ensemble Average
    st.sidebar.success(f"Ensemble Average: ${np.mean(list(preds.values())):,.2f}")

# ----------------- Analysis & Graphs -----------------
st.subheader("📊 Salary Insights")

# Average Salary by Job Title
avg_job = df_raw.groupby("job_title")["salary_in_usd"].mean().reset_index()
fig1 = px.bar(avg_job, x="job_title", y="salary_in_usd", title="Average Salary by Job Title")
st.plotly_chart(fig1, use_container_width=True)

# Average Salary by Experience Level
avg_exp = df_raw.groupby("experience_level")["salary_in_usd"].mean().reset_index()
fig2 = px.bar(avg_exp, x="experience_level", y="salary_in_usd", title="Average Salary by Experience Level")
st.plotly_chart(fig2, use_container_width=True)

# Salary Distribution
fig3 = px.box(df_raw, x="experience_level", y="salary_in_usd", color="experience_level",
              title="Salary Distribution by Experience Level")
st.plotly_chart(fig3, use_container_width=True)

# Remote Ratio vs Salary
fig4 = px.scatter(df_raw, x="remote_ratio", y="salary_in_usd", color="experience_level",
                  title="Remote Ratio vs Salary")
st.plotly_chart(fig4, use_container_width=True)

# Model Performance
st.subheader("Model Performance")
st.dataframe(perf_df, use_container_width=True)

