"""
Telecom Churn Intelligence: Predict, Prevent and Retain
Run from the repo root:  streamlit run app/app.py
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from joblib import load

# ----------------------------------------------------------------------------
# Page setup and styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK, SLATE = "#14213D", "#5B6577"
RED, AMBER, GREEN, TEAL = "#C8372D", "#D99A1E", "#2E7D5B", "#1F6F78"
CURRENCY = "$"  # Telco dataset charges are in USD

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
      .hero {{ background: {INK}; color: #fff; padding: 1.4rem 1.8rem;
               border-radius: 10px; margin-bottom: 1.2rem; }}
      .hero h1 {{ color: #fff; font-size: 1.7rem; margin: 0 0 .25rem 0; padding: 0; }}
      .hero p {{ color: #C9D1E0; margin: 0; font-size: .98rem; }}
      .meter {{ position: relative; height: 14px; border-radius: 7px; margin: .6rem 0 1.4rem 0; }}
      .meter .pin {{ position: absolute; top: -6px; width: 4px; height: 26px;
                     background: {INK}; border-radius: 2px; }}
      .meter-scale {{ display: flex; justify-content: space-between; color: {SLATE};
                      font-size: .78rem; margin-top: -1.1rem; margin-bottom: 1rem; }}
      .tier {{ display: inline-block; padding: .15rem .7rem; border-radius: 999px;
               color: #fff; font-weight: 600; font-size: .9rem; }}
      div[data-testid="stMetricValue"] {{ font-size: 1.7rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Schema, paths and constants
# ----------------------------------------------------------------------------
NUMERIC = ["MonthlyCharges", "TotalCharges", "tenure"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
          "StreamingTV", "StreamingMovies"]
MODEL_FILES = {
    "Logistic Regression": "logistic_model.pkl",
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl",
}
# Used only if models/metrics.json is missing (values from the notebook run)
FALLBACK_METRICS = {
    "Logistic Regression": {"ROC AUC": 0.85, "Recall": 0.79, "Precision": 0.54, "F1": 0.64},
    "Random Forest": {"ROC AUC": 0.85, "Recall": 0.72, "Precision": 0.57, "F1": 0.63},
    "XGBoost": {"ROC AUC": 0.85, "Recall": 0.75, "Precision": 0.55, "F1": 0.63},
}
DEFAULT_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 80.0,
}


def find_dir(name: str) -> Path:
    here = Path(__file__).resolve().parent
    for base in (here, here.parent):
        if (base / name).exists():
            return base / name
    return here.parent / name


MODELS_DIR, DATA_DIR = find_dir("models"), find_dir("data")


# ----------------------------------------------------------------------------
# Loading and scoring helpers
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models() -> dict:
    models = {}
    for name, file in MODEL_FILES.items():
        path = MODELS_DIR / file
        if path.exists():
            models[name] = load(path)
    return models


@st.cache_data(show_spinner="Loading customer data...")
def load_data():
    path = DATA_DIR / "Telco_Customer_Churn.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df["Churned"] = (df["Churn"] == "Yes").astype(int)
    return df


@st.cache_data
def load_metrics() -> tuple[pd.DataFrame, bool]:
    path = MODELS_DIR / "metrics.json"
    if path.exists():
        try:
            return pd.DataFrame(json.loads(path.read_text())).T, True
        except Exception:
            pass
    return pd.DataFrame(FALLBACK_METRICS).T, False


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def split_pipeline(pipe):
    return pipe.steps[0][1], pipe.steps[-1][1]  # preprocessor, estimator


def score(model_name: str, X: pd.DataFrame) -> np.ndarray:
    return load_models()[model_name].predict_proba(X[FEATURES])[:, 1]


@st.cache_data(show_spinner="Scoring customers...")
def score_portfolio(model_name: str) -> pd.DataFrame | None:
    df = load_data()
    if df is None:
        return None
    out = df.copy()
    out["Risk score"] = score(model_name, out)
    return out


def tier_of(p: float, hi: float, med: float) -> tuple[str, str]:
    if p >= hi:
        return "High", RED
    if p >= med:
        return "Medium", AMBER
    return "Low", GREEN


def tier_series(p: pd.Series, hi: float, med: float) -> pd.Series:
    return pd.cut(p, [-0.01, med, hi, 1.01], labels=["Low", "Medium", "High"], right=False)


# ----------------------------------------------------------------------------
# Explainability (SHAP) per selected model
# ----------------------------------------------------------------------------
FRIENDLY = {"tenure": "Tenure", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges"}


def base_name(feature: str) -> str:
    name = feature.split("__", 1)[-1]
    return name.split("_", 1)[0]


def pretty(feature: str) -> str:
    name = feature.split("__", 1)[-1]
    if "_" in name:
        head, tail = name.split("_", 1)
        return f"{re.sub(r'(?<!^)(?=[A-Z])', ' ', head)}: {tail}"
    return FRIENDLY.get(name, re.sub(r"(?<!^)(?=[A-Z])", " ", name))


def pretty_group(name: str) -> str:
    return FRIENDLY.get(name, re.sub(r"(?<!^)(?=[A-Z])", " ", name))


def class1(sv):
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


@st.cache_resource(show_spinner="Preparing explanations...")
def get_explainer(model_name: str):
    pre, est = split_pipeline(load_models()[model_name])
    try:
        if model_name == "Logistic Regression":
            df = load_data()
            if df is None:
                return None
            bg = dense(pre.transform(df[FEATURES].sample(min(200, len(df)), random_state=42)))
            return shap.LinearExplainer(est, bg)
        return shap.TreeExplainer(est)
    except Exception:
        return None


def explain_rows(model_name: str, X: pd.DataFrame):
    """Return (shap matrix, feature names) or (None, None)."""
    explainer = get_explainer(model_name)
    if explainer is None:
        return None, None
    pre, _ = split_pipeline(load_models()[model_name])
    Xt = dense(pre.transform(X[FEATURES]))
    try:
        return class1(explainer.shap_values(Xt)), list(pre.get_feature_names_out())
    except Exception:
        return None, None


def bar_chart(values: pd.Series, colors, title: str, xlabel: str, height: float = 3.6):
    fig, ax = plt.subplots(figsize=(7, height))
    ax.barh(values.index, values.values, color=colors)
    ax.invert_yaxis()
    ax.axvline(0, color=SLATE, lw=0.8)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=SLATE, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=SLATE, labelsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Retention playbook (rule-based recommendations)
# ----------------------------------------------------------------------------
def recommend(c: dict) -> list[tuple[str, str]]:
    recs = []
    if c["Contract"] == "Month-to-month":
        recs.append(("Offer a 12-month contract at a loyalty discount",
                     "Month-to-month customers carry the highest churn risk."))
    if c["InternetService"] != "No" and (c["TechSupport"] == "No" or c["OnlineSecurity"] == "No"):
        recs.append(("Bundle Tech Support and Online Security free for 3 months",
                     "Customers without these services churn more often."))
    if c["tenure"] <= 12:
        recs.append(("Enroll in a 90-day onboarding check-in program",
                     "Early-tenure customers are the most likely to leave."))
    if c["InternetService"] == "Fiber optic" and c["MonthlyCharges"] >= 80:
        recs.append(("Review the plan for a better price-to-value fit",
                     "High-bill fiber customers are price-sensitive."))
    if c["PaymentMethod"] == "Electronic check":
        recs.append(("Incentivize a switch to automatic payment",
                     "Electronic-check payers show higher churn."))
    if not recs:
        recs.append(("Keep in standard engagement",
                     "No major churn signals in the current profile."))
    return recs


# ----------------------------------------------------------------------------
# Guard: models must exist
# ----------------------------------------------------------------------------
MODELS = load_models()
if not MODELS:
    st.error(
        f"No model files found in `{MODELS_DIR}`. Save `logistic_model.pkl`, "
        "`rf_model.pkl` and `xgb_model.pkl` there and reload."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Sidebar: model and business assumptions
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Prediction model", list(MODELS), index=list(MODELS).index("Random Forest") if "Random Forest" in MODELS else 0)

    st.subheader("Risk tiers")
    hi = st.slider("High risk from", 0.30, 0.90, 0.60, 0.05)
    med = st.slider("Medium risk from", 0.10, hi - 0.05, min(0.30, hi - 0.05), 0.05)

    st.subheader("Retention economics")
    offer_cost = st.number_input(f"Offer cost per customer ({CURRENCY})", 0.0, 500.0, 40.0, 5.0)
    save_rate = st.slider("Share of targeted churners saved", 0.05, 0.80, 0.25, 0.05, format="%.2f")
    lifetime = st.slider("Months of revenue kept per save", 3, 36, 12)
    st.caption("Scores come from models trained with class weighting, so they rank risk "
               "well but are not exact probabilities.")

st.markdown(
    """
    <div class="hero">
      <h1>Telecom Churn Intelligence</h1>
      <p>Predict who is likely to leave, understand why, and decide where a retention offer pays back.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_cust, tab_port, tab_roi, tab_model = st.tabs(
    ["Customer risk", "Portfolio overview", "Retention ROI", "Model insights"]
)

# ----------------------------------------------------------------------------
# Tab 1: single-customer assessment
# ----------------------------------------------------------------------------
with tab_cust:
    st.session_state.setdefault("customer", DEFAULT_CUSTOMER.copy())
    cur = st.session_state["customer"]

    form_col, result_col = st.columns([4, 6], gap="large")

    with form_col:
        st.subheader("Customer profile")
        with st.form("customer_form"):
            with st.expander("Account", expanded=True):
                tenure = st.slider("Tenure (months)", 0, 72, int(cur["tenure"]))
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"],
                                        ["Month-to-month", "One year", "Two year"].index(cur["Contract"]))
                payment = st.selectbox(
                    "Payment method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                     "Credit card (automatic)"].index(cur["PaymentMethod"]))
                paperless = st.selectbox("Paperless billing", ["Yes", "No"],
                                         ["Yes", "No"].index(cur["PaperlessBilling"]))
                monthly = st.number_input(f"Monthly charges ({CURRENCY})", 15.0, 200.0, float(cur["MonthlyCharges"]), 1.0)
            with st.expander("Services", expanded=True):
                phone = st.selectbox("Phone service", ["Yes", "No"], ["Yes", "No"].index(cur["PhoneService"]))
                lines = st.selectbox("Multiple lines", ["Yes", "No"], 1 if cur["MultipleLines"] != "Yes" else 0)
                internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"],
                                        ["DSL", "Fiber optic", "No"].index(cur["InternetService"]))
                addon_vals = {}
                for a in ADDONS:
                    label = re.sub(r"(?<!^)(?=[A-Z])", " ", a)
                    addon_vals[a] = st.selectbox(label, ["Yes", "No"], 0 if cur[a] == "Yes" else 1)
            with st.expander("Demographics"):
                gender = st.selectbox("Gender", ["Female", "Male"], ["Female", "Male"].index(cur["gender"]))
                senior = st.selectbox("Senior citizen", [0, 1], int(cur["SeniorCitizen"]))
                partner = st.selectbox("Partner", ["Yes", "No"], ["Yes", "No"].index(cur["Partner"]))
                dependents = st.selectbox("Dependents", ["Yes", "No"], ["Yes", "No"].index(cur["Dependents"]))
            submitted = st.form_submit_button("Assess customer", type="primary", width="stretch")

        if submitted:
            cust = {
                "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
                "tenure": tenure, "PhoneService": phone,
                "MultipleLines": "No phone service" if phone == "No" else lines,
                "InternetService": internet, "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment, "MonthlyCharges": monthly,
            }
            for a in ADDONS:
                cust[a] = "No internet service" if internet == "No" else addon_vals[a]
            st.session_state["customer"] = cust
            cur = cust

    # Build the model input (TotalCharges is estimated from tenure and bill)
    row = dict(cur)
    row["TotalCharges"] = round(row["tenure"] * row["MonthlyCharges"], 2)
    X1 = pd.DataFrame([row])[FEATURES]
    p = float(score(model_name, X1)[0])
    tier, color = tier_of(p, hi, med)
    monthly_bill = row["MonthlyCharges"]
    net_benefit = save_rate * p * monthly_bill * lifetime - offer_cost

    with result_col:
        st.subheader("Risk assessment")
        k1, k2, k3, k4 = st.columns(4)
        with k1.container(border=True):
            st.metric("Risk score", f"{p * 100:.0f}%")
        with k2.container(border=True):
            st.markdown("Risk tier")
            st.markdown(f'<span class="tier" style="background:{color}">{tier}</span>', unsafe_allow_html=True)
        with k3.container(border=True):
            st.metric("Annual revenue at stake", f"{CURRENCY}{monthly_bill * 12:,.0f}")
        with k4.container(border=True):
            st.metric("Offer net value", f"{CURRENCY}{net_benefit:,.0f}",
                      help="Save rate x risk score x monthly bill x months kept, minus offer cost.")

        m, h = med * 100, hi * 100
        st.markdown(
            f"""
            <div class="meter" style="background:linear-gradient(90deg,
                 {GREEN} 0%, {GREEN} {m}%, {AMBER} {m}%, {AMBER} {h}%, {RED} {h}%, {RED} 100%);">
              <div class="pin" style="left:calc({p * 100:.1f}% - 2px)"></div>
            </div>
            <div class="meter-scale"><span>Low</span><span>Medium from {m:.0f}%</span><span>High from {h:.0f}%</span></div>
            """,
            unsafe_allow_html=True,
        )

        if tier == "Low":
            st.success("Low churn risk. Keep the customer in standard engagement.")
        elif net_benefit > 0:
            st.info(f"A retention offer is worth making: expected net value of {CURRENCY}{net_benefit:,.0f} "
                    "under the sidebar assumptions.")
        else:
            st.warning("A retention offer is unlikely to pay back under the sidebar assumptions. "
                       "Consider a lower-cost action.")

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**What is driving this score**")
            sv, names = explain_rows(model_name, X1)
            if sv is None:
                st.caption("Explanations for this model need the training data file in the data folder.")
            else:
                s = pd.Series(sv[0], index=[pretty(n) for n in names])
                top = s.reindex(s.abs().sort_values(ascending=False).index).head(7)
                fig = bar_chart(top, [RED if v > 0 else GREEN for v in top.values],
                                "Top factors", "Red raises risk, green lowers it")
                st.pyplot(fig, width="stretch")
                plt.close(fig)
        with right:
            st.markdown("**Recommended retention actions**")
            for action, why in recommend(row):
                with st.container(border=True):
                    st.markdown(f"**{action}**")
                    st.caption(why)

# ----------------------------------------------------------------------------
# Tab 2: portfolio overview
# ----------------------------------------------------------------------------
scored = score_portfolio(model_name)

with tab_port:
    if scored is None:
        st.info(f"Add `Telco_Customer_Churn.csv` to `{DATA_DIR}` to see the portfolio view.")
    else:
        scored = scored.copy()
        scored["Tier"] = tier_series(scored["Risk score"], hi, med)
        churners = scored[scored["Churned"] == 1]

        k1, k2, k3, k4 = st.columns(4)
        with k1.container(border=True):
            st.metric("Customers", f"{len(scored):,}")
        with k2.container(border=True):
            st.metric("Churn rate", f"{scored['Churned'].mean() * 100:.1f}%")
        with k3.container(border=True):
            st.metric("Monthly revenue lost to churn", f"{CURRENCY}{churners['MonthlyCharges'].sum():,.0f}")
        with k4.container(border=True):
            st.metric("Median tenure of churners", f"{churners['tenure'].median():.0f} months")

        st.subheader("Where churn concentrates")
        scored["Tenure band"] = pd.cut(scored["tenure"], [-1, 12, 24, 48, 72],
                                       labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"])
        segs = [("Contract", "Contract"), ("Tenure band", "Tenure"),
                ("InternetService", "Internet service"), ("PaymentMethod", "Payment method")]
        cols = st.columns(2, gap="large")
        for i, (col, label) in enumerate(segs):
            rate = (scored.groupby(col, observed=True)["Churned"].mean() * 100).sort_values(ascending=False)
            fig = bar_chart(rate.round(1), TEAL, f"Churn rate by {label.lower()}", "Churn rate (%)", 2.6)
            with cols[i % 2]:
                st.pyplot(fig, width="stretch")
            plt.close(fig)

        st.subheader("Customers to contact first")
        st.caption("Ranked by risk score and monthly bill. Scores are computed on the historical "
                   "dataset the models were trained on, so use fresh data for live campaigns.")
        queue = (scored[scored["Tier"] == "High"]
                 .sort_values(["Risk score", "MonthlyCharges"], ascending=False)
                 [["customerID", "Risk score", "MonthlyCharges", "tenure", "Contract",
                   "InternetService", "PaymentMethod"]]
                 .rename(columns={"MonthlyCharges": "Monthly charges", "tenure": "Tenure"}))
        st.dataframe(
            queue.head(25), hide_index=True, width="stretch",
            column_config={"Risk score": st.column_config.ProgressColumn(
                "Risk score", min_value=0.0, max_value=1.0, format="%.2f")},
        )
        st.download_button("Download full high-risk list (CSV)", queue.to_csv(index=False).encode(),
                           "high_risk_customers.csv", "text/csv")

# ----------------------------------------------------------------------------
# Tab 3: retention ROI simulator
# ----------------------------------------------------------------------------
with tab_roi:
    if scored is None:
        st.info(f"Add `Telco_Customer_Churn.csv` to `{DATA_DIR}` to run the ROI simulator.")
    else:
        st.subheader("Campaign simulator")
        st.caption("Choose which risk tiers receive an offer. Churn counts are the real outcomes in the "
                   "dataset; cost and save assumptions come from the sidebar.")
        targets = st.multiselect("Tiers to target", ["High", "Medium", "Low"], default=["High"])
        seg = scored[scored["Tier"].isin(targets)]

        if seg.empty:
            st.warning("Select at least one tier.")
        else:
            contacted = len(seg)
            churners_in = int(seg["Churned"].sum())
            saved = churners_in * save_rate
            avg_bill = seg.loc[seg["Churned"] == 1, "MonthlyCharges"].mean() if churners_in else 0.0
            revenue_saved = saved * avg_bill * lifetime
            cost = contacted * offer_cost
            net = revenue_saved - cost

            k1, k2, k3, k4 = st.columns(4)
            with k1.container(border=True):
                st.metric("Customers contacted", f"{contacted:,}")
            with k2.container(border=True):
                st.metric("Churners reached", f"{churners_in:,}",
                          f"{churners_in / max(scored['Churned'].sum(), 1) * 100:.0f}% of all churners",
                          delta_color="off")
            with k3.container(border=True):
                st.metric("Campaign cost", f"{CURRENCY}{cost:,.0f}")
            with k4.container(border=True):
                st.metric("Net benefit", f"{CURRENCY}{net:,.0f}", f"{revenue_saved:,.0f} revenue kept",
                          delta_color="off")

            if net > 0:
                st.success(f"Targeting {', '.join(targets).lower()} risk customers returns about "
                           f"{CURRENCY}{net:,.0f} after offer costs.")
            else:
                st.warning("This campaign does not pay back under the current assumptions. "
                           "Lower the offer cost, tighten the target tiers, or raise the save rate in the sidebar.")

        st.subheader("Net benefit by targeting rule")
        rows = []
        for t in np.arange(0.2, 0.85, 0.05):
            s = scored[scored["Risk score"] >= t]
            ch = s["Churned"].sum()
            bill = s.loc[s["Churned"] == 1, "MonthlyCharges"].mean() if ch else 0.0
            rows.append({"Risk score cutoff": round(t, 2),
                         "Net benefit": ch * save_rate * bill * lifetime - len(s) * offer_cost})
        curve = pd.DataFrame(rows).set_index("Risk score cutoff")
        best = curve["Net benefit"].idxmax()
        st.line_chart(curve, color=TEAL, height=260)
        st.caption(f"On this data the highest net benefit comes from contacting customers with a risk "
                   f"score of {best:.2f} or higher.")

# ----------------------------------------------------------------------------
# Tab 4: model insights
# ----------------------------------------------------------------------------
with tab_model:
    metrics, from_file = load_metrics()
    st.subheader("Model comparison")
    st.dataframe(
        metrics.style.format("{:.2f}").highlight_max(axis=0, color="#DCEFE6"),
        width="stretch",
    )
    st.caption("Metrics on the held-out test set." if from_file else
               "Showing notebook results. Save `models/metrics.json` from the notebook to keep this table in sync.")

    st.subheader(f"What drives churn ({model_name})")
    df_all = load_data()
    if df_all is None:
        st.info(f"Add `Telco_Customer_Churn.csv` to `{DATA_DIR}` to see global drivers.")
    else:
        sample = df_all[FEATURES].sample(min(400, len(df_all)), random_state=42)
        sv, names = explain_rows(model_name, sample)
        if sv is None:
            st.info("Explanations are not available for this model in the current environment.")
        else:
            imp = pd.Series(np.abs(sv).mean(axis=0), index=[base_name(n) for n in names])
            imp = imp.groupby(level=0).sum().sort_values(ascending=False).head(10)
            imp.index = [pretty_group(i) for i in imp.index]
            fig = bar_chart(imp, TEAL, "Top 10 churn drivers", "Average impact on the risk score", 4.0)
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.caption("Impact is the average absolute SHAP value across a real sample of customers, "
                       "summed across the categories of each feature.")

    with st.expander("How to read these results"):
        st.markdown(
            """
            - **Contract type and tenure** are consistently the strongest signals: new customers on
              month-to-month plans are the highest-risk group.
            - **Service add-ons** such as Tech Support and Online Security are associated with lower churn.
            - **Fiber optic and electronic-check customers** show higher churn, which often points to
              price sensitivity or service quality.
            - Use the risk score to **rank** customers, and use the ROI tab to decide **how many** to contact.
            """
        )
