import streamlit as st
import pandas as pd
import plotly.express as px

# Page config

st.set_page_config(page_title="Adult Census — Plotly Express Studio", layout="wide")

st.title("📊 Adult Census Income — Plotly Express Studio")
st.caption(
    "An interactive gallery of 10+ Plotly Express visualization scenarios "
    "built on the UCI Adult (Census Income) dataset."
)

# Load data

@st.cache_data
def load_data():
    df = pd.read_csv("adult.csv")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df["income"] = df["income"].str.strip()
    return df

df = load_data()

# ----------------------------------------------------------------------
# Sidebar filters (make it feel like a "studio", not a static report)
# ----------------------------------------------------------------------
st.sidebar.header("🔎 Filters")

age_range = st.sidebar.slider(
    "Age range", int(df.age.min()), int(df.age.max()),
    (int(df.age.min()), int(df.age.max()))
)
genders = st.sidebar.multiselect(
    "Gender", sorted(df.Gender.dropna().unique()), default=sorted(df.Gender.dropna().unique())
)
incomes = st.sidebar.multiselect(
    "Income bracket", sorted(df.income.dropna().unique()), default=sorted(df.income.dropna().unique())
)

filtered = df[
    (df.age.between(*age_range)) &
    (df.Gender.isin(genders)) &
    (df.income.isin(incomes))
]

st.sidebar.markdown(f"**Rows after filtering:** {len(filtered):,} / {len(df):,}")

# Scenario layout helper

def scenario(number, title, description):
    st.subheader(f"{number}. {title}")
    st.caption(description)

# SCENARIO 1 — Histogram

scenario(1, "Age Distribution by Income", "Histogram of age, split and colored by income bracket.")
fig1 = px.histogram(
    filtered, x="age", color="income", nbins=30, barmode="overlay", opacity=0.7,
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={"age": "Age", "income": "Income"}
)
st.plotly_chart(fig1, use_container_width=True)

# SCENARIO 2 — Box plot

scenario(2, "Hours Worked per Week by Education", "Box plot showing spread of weekly hours across education levels.")
fig2 = px.box(
    filtered, x="education", y="hours.per.week", color="income",
    category_orders={"education": sorted(filtered.education.unique())},
    labels={"hours.per.week": "Hours / Week", "education": "Education"}
)
fig2.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig2, use_container_width=True)

# SCENARIO 3 — Bar chart (counts)

scenario(3, "Income Split by Workclass", "Stacked bar chart of income bracket counts within each workclass.")
work_income = filtered.dropna(subset=["workclass"]).groupby(["workclass", "income"]).size().reset_index(name="count")
fig3 = px.bar(
    work_income, x="workclass", y="count", color="income", barmode="stack",
    labels={"count": "Number of People", "workclass": "Workclass"}
)
fig3.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig3, use_container_width=True)

# SCENARIO 4 — Scatter plot

scenario(4, "Age vs. Hours per Week", "Scatter plot colored by income and sized by education level (numeric).")
fig4 = px.scatter(
    filtered, x="age", y="hours.per.week", color="income", size="education.num",
    hover_data=["occupation", "education"], opacity=0.5,
    labels={"hours.per.week": "Hours / Week", "age": "Age"}
)
st.plotly_chart(fig4, use_container_width=True)

# SCENARIO 5 — Pie chart

scenario(5, "Gender Distribution", "Pie chart of the overall gender split in the filtered data.")
fig5 = px.pie(filtered, names="Gender", color="Gender", hole=0.4)
st.plotly_chart(fig5, use_container_width=True)

# SCENARIO 6 — Violin plot

scenario(6, "Age Distribution by Marital Status", "Violin plot revealing the shape of the age distribution per marital status group.")
fig6 = px.violin(
    filtered, x="marital.status", y="age", color="income", box=True, points=False,
    labels={"marital.status": "Marital Status"}
)
fig6.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig6, use_container_width=True)

# SCENARIO 7 — Sunburst

scenario(7, "Income Hierarchy: Income → Education → Gender", "Sunburst chart drilling from income bracket down to education and gender.")
sun_df = filtered.dropna(subset=["education"])
income_color_map = {"<=50K": "#636EFA", ">50K": "#EF553B"}
fig7 = px.sunburst(
    sun_df, path=["income", "education", "Gender"], color="income",
    color_discrete_map=income_color_map
)
st.plotly_chart(fig7, use_container_width=True)

# SCENARIO 8 — Correlation heatmap

scenario(8, "Correlation Heatmap of Numeric Features", "Heatmap of pairwise correlations between numeric columns.")
numeric_cols = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]
corr = filtered[numeric_cols].corr()
fig8 = px.imshow(
    corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    labels=dict(color="Correlation")
)
st.plotly_chart(fig8, use_container_width=True)

# SCENARIO 9 — Grouped bar (aggregated average)

scenario(9, "Average Weekly Hours by Occupation", "Bar chart of mean hours-per-week for each occupation.")
occ_avg = filtered.dropna(subset=["occupation"]).groupby("occupation", as_index=False)["hours.per.week"].mean()
occ_avg = occ_avg.sort_values("hours.per.week", ascending=False)
fig9 = px.bar(
    occ_avg, x="occupation", y="hours.per.week", color="hours.per.week",
    color_continuous_scale="Viridis", labels={"hours.per.week": "Avg Hours / Week"}
)
fig9.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig9, use_container_width=True)

# SCENARIO 10 — Treemap

scenario(10, "Population Breakdown: Race → Workclass → Income", "Treemap showing relative group sizes across race, workclass, and income. Note: block size reflects real sample counts — this dataset is heavily skewed toward 'White' respondents, so that group's block will always dominate; this is a property of the data, not the chart.")
tree_df = filtered.dropna(subset=["workclass"])
fig10 = px.treemap(
    tree_df, path=["race", "workclass", "income"], color="income",
    color_discrete_map=income_color_map
)
st.plotly_chart(fig10, use_container_width=True)

st.markdown("**Alternative view — same data, normalized so small groups are visible:**")
race_income_pct = (
    tree_df.groupby(["race", "income"]).size().reset_index(name="count")
)
race_income_pct["pct"] = race_income_pct.groupby("race")["count"].transform(lambda x: x / x.sum() * 100)
fig10b = px.bar(
    race_income_pct, x="race", y="pct", color="income", barmode="stack",
    color_discrete_map=income_color_map,
    labels={"pct": "% within Race Group", "race": "Race"}
)
fig10b.update_layout(xaxis_tickangle=-20)
st.plotly_chart(fig10b, use_container_width=True)

# SCENARIO 11 (bonus) — Density heatmap

scenario(11, "Density Heatmap: Age vs. Education Years", "2D density heatmap showing where age and years-of-education combinations concentrate.")
fig11 = px.density_heatmap(
    filtered, x="age", y="education.num", nbinsx=30, nbinsy=16,
    color_continuous_scale="Blues", labels={"education.num": "Education (years)"}
)
st.plotly_chart(fig11, use_container_width=True)


st.markdown("---")
st.caption("Built with Streamlit + Plotly Express · Dataset: UCI Adult / Census Income")
