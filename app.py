import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Liver Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #e94560;
    }
    .main-header h1 { color: #e94560; font-size: 2.5rem; margin: 0; }
    .main-header p  { color: #a0aec0; margin: 0.5rem 0 0 0; font-size: 1rem; }

    .result-positive {
        background: linear-gradient(135deg, #ff4757, #ff6b81);
        color: white; padding: 1.5rem; border-radius: 12px;
        text-align: center; font-size: 1.4rem; font-weight: bold;
        margin: 1rem 0; box-shadow: 0 4px 15px rgba(255,71,87,0.4);
    }
    .result-negative {
        background: linear-gradient(135deg, #2ed573, #1e90ff);
        color: white; padding: 1.5rem; border-radius: 12px;
        text-align: center; font-size: 1.4rem; font-weight: bold;
        margin: 1rem 0; box-shadow: 0 4px 15px rgba(46,213,115,0.4);
    }
    .metric-card {
        background: #1a1a2e; border: 1px solid #0f3460;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #e94560 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load & Train Model ────────────────────────────────────────────────────────
@st.cache_resource
def load_data_and_train():
    df = pd.read_csv("liver_disease_2000.csv")
    X  = df.drop("liver_disease", axis=1)
    y  = df["liver_disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=20)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    smote = SMOTE(random_state=20)
    X_res, y_res = smote.fit_resample(X_train_sc, y_train)

    # All models from notebook
    models = {
        "Logistic Regression":  LogisticRegression(max_iter=1000),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost":              XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42),
        "AdaBoost":             AdaBoostClassifier(n_estimators=100, random_state=42),
        "Decision Tree":        DecisionTreeClassifier(random_state=42),
        "KNN":                  KNeighborsClassifier(n_neighbors=7),
        "SVM (RBF)":            SVC(kernel="rbf", probability=True),
    }

    trained = {}
    accs    = {}
    for name, m in models.items():
        m.fit(X_res, y_res)
        accs[name] = round(accuracy_score(y_test, m.predict(X_test_sc)) * 100, 2)
        trained[name] = m

    # Voting Classifier (best model)
    voting = VotingClassifier(
        estimators=[
            ("lr",  LogisticRegression(max_iter=1000)),
            ("rf",  RandomForestClassifier(n_estimators=100, random_state=42)),
            ("xgb", XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42)),
        ],
        voting="soft"
    )
    voting.fit(X_res, y_res)
    accs["Voting Classifier"] = round(accuracy_score(y_test, voting.predict(X_test_sc)) * 100, 2)
    trained["Voting Classifier"] = voting

    return df, X, y, scaler, trained, accs, X_test_sc, y_test, X_res, y_res

df, X, y, scaler, trained_models, model_accs, X_test_sc, y_test, X_res, y_res = load_data_and_train()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🫀 Liver Disease Predictor</h1>
    <p>ML-powered prediction using Logistic Regression · Random Forest · XGBoost · SVM · KNN · Voting Classifier</p>
    <p style="color:#e94560; font-size:0.85rem;">Built by Akula Dhanalaxmi · SR University · B.Tech CSE</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔬 Patient Input")
st.sidebar.markdown("---")

age                    = st.sidebar.slider("Age (years)",                    18, 84, 45)
total_bilirubin        = st.sidebar.slider("Total Bilirubin (mg/dL)",        0.2, 20.0, 10.0, 0.1)
direct_bilirubin       = st.sidebar.slider("Direct Bilirubin (mg/dL)",       0.0, 10.0, 3.0, 0.1)
alkaline_phosphotase   = st.sidebar.slider("Alkaline Phosphotase (IU/L)",    40,  300,  150)
alanine_amino          = st.sidebar.slider("Alanine Aminotransferase (U/L)", 5,   250,  60)
aspartate_amino        = st.sidebar.slider("Aspartate Aminotransferase (U/L)",5,  250,  60)
total_proteins         = st.sidebar.slider("Total Proteins (g/dL)",          4.0, 9.0,  6.5, 0.1)
albumin                = st.sidebar.slider("Albumin (g/dL)",                 2.0, 5.5,  3.5, 0.1)
albumin_globulin_ratio = st.sidebar.slider("Albumin/Globulin Ratio",         0.5, 2.5,  1.2, 0.1)

st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox(
    "🤖 Select Model",
    list(trained_models.keys()),
    index=list(trained_models.keys()).index("Voting Classifier")
)
predict_btn = st.sidebar.button("🔍 Predict", use_container_width=True, type="primary")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Model Comparison", "📈 Data Insights", "📋 About"])

# ══════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════
with tab1:
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown("### Patient Summary")
        summary_df = pd.DataFrame({
            "Feature": [
                "Age", "Total Bilirubin", "Direct Bilirubin",
                "Alkaline Phosphotase", "Alanine Aminotransferase",
                "Aspartate Aminotransferase", "Total Proteins",
                "Albumin", "Albumin/Globulin Ratio"
            ],
            "Value": [
                age, total_bilirubin, direct_bilirubin,
                alkaline_phosphotase, alanine_amino,
                aspartate_amino, total_proteins,
                albumin, albumin_globulin_ratio
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        if predict_btn:
            input_data = np.array([[
                age, total_bilirubin, direct_bilirubin,
                alkaline_phosphotase, alanine_amino,
                aspartate_amino, total_proteins,
                albumin, albumin_globulin_ratio
            ]])
            input_scaled = scaler.transform(input_data)
            model        = trained_models[model_choice]
            prediction   = model.predict(input_scaled)[0]
            proba        = model.predict_proba(input_scaled)[0]

            if prediction == 1:
                st.markdown(f"""
                <div class="result-positive">
                    🔴 Liver Disease Detected<br>
                    <small>Confidence: {proba[1]*100:.1f}%</small>
                </div>""", unsafe_allow_html=True)
                st.warning("⚠️ This is a screening tool only. Please consult a qualified hepatologist.")
            else:
                st.markdown(f"""
                <div class="result-negative">
                    🟢 No Liver Disease Detected<br>
                    <small>Confidence: {proba[0]*100:.1f}%</small>
                </div>""", unsafe_allow_html=True)
                st.info("✅ Results look normal. Regular health checkups are still recommended.")

            # Probability bar
            st.markdown("#### Prediction Probability")
            fig, ax = plt.subplots(figsize=(7, 1.2))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            ax.barh(["No Disease"], [proba[0]], color="#2ed573", height=0.4)
            ax.barh(["Disease"],    [proba[1]], color="#ff4757", height=0.4)
            ax.set_xlim(0, 1)
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_visible(False)
            for i, (label, val) in enumerate(zip(["No Disease","Disease"], proba)):
                ax.text(val + 0.01, i, f"{val*100:.1f}%", va='center', color='white', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        else:
            st.info("👈 Adjust the patient values in the sidebar and click **Predict**")

    # Quick model metrics
    st.markdown("---")
    st.markdown("### Quick Model Accuracy Overview")
    cols = st.columns(len(trained_models))
    for col, (name, acc) in zip(cols, model_accs.items()):
        short = name.replace("Voting Classifier","Voting").replace("Gradient Boosting","GBM").replace("Logistic Regression","LR").replace("Random Forest","RF").replace("Decision Tree","DT")
        col.metric(short, f"{acc}%")

# ══════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model Accuracy Comparison (After SMOTE)")

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1a1a2e')
        names  = list(model_accs.keys())
        values = list(model_accs.values())
        colors = ['#e94560' if v == max(values) else '#0f3460' for v in values]
        bars   = ax.bar(names, values, color=colors, edgecolor='#16213e', linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{val}%", ha='center', va='bottom', color='white', fontsize=8, fontweight='bold')
        ax.set_ylabel("Accuracy (%)", color='white')
        ax.set_ylim(0, 100)
        ax.tick_params(colors='white', labelsize=7.5)
        plt.xticks(rotation=30, ha='right')
        for spine in ax.spines.values(): spine.set_color('#0f3460')
        ax.yaxis.label.set_color('white')
        ax.set_title("Model Accuracy Comparison", color='white', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        # ROC Curves
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1a1a2e')
        palette = ['#e94560','#0f3460','#2ed573','#ffa502','#ff6b81','#1e90ff','#a29bfe','#fd79a8','#00b894']
        for (name, model), color in zip(trained_models.items(), palette):
            try:
                probs = model.predict_proba(X_test_sc)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                roc_auc = auc(fpr, tpr)
                short = name.replace("Voting Classifier","Voting").replace("Gradient Boosting","GBM").replace("Logistic Regression","LR").replace("Random Forest","RF").replace("Decision Tree","DT")
                ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{short} ({roc_auc:.2f})")
            except: pass
        ax.plot([0,1],[0,1],'w--', lw=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate", color='white')
        ax.set_ylabel("True Positive Rate", color='white')
        ax.set_title("ROC Curves — All Models", color='white', fontweight='bold')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#0f3460')
        ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Confusion Matrix for selected model
    st.markdown(f"### Confusion Matrix — {model_choice}")
    selected_model = trained_models[model_choice]
    y_pred_sel     = selected_model.predict(X_test_sc)
    cm             = confusion_matrix(y_test, y_pred_sel)

    col3, col4 = st.columns([1, 2])
    with col3:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        fig.patch.set_facecolor('#0e1117')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Disease','Disease'],
                    yticklabels=['No Disease','Disease'],
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(f"Confusion Matrix", color='white', fontweight='bold')
        ax.set_xlabel("Predicted", color='white')
        ax.set_ylabel("Actual", color='white')
        ax.tick_params(colors='white')
        fig.patch.set_facecolor('#0e1117')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        report = classification_report(y_test, y_pred_sel, target_names=["No Disease","Disease"], output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", "2,000")
    col2.metric("Features", "9")
    col3.metric("Liver Disease", f"{int(y.sum())} ({y.mean()*100:.1f}%)")
    col4.metric("No Disease", f"{int((y==0).sum())} ({(y==0).mean()*100:.1f}%)")

    col1, col2 = st.columns(2)

    with col1:
        # Target distribution
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1a1a2e')
        counts = y.value_counts()
        wedges, texts, autotexts = ax.pie(
            counts, labels=['No Disease','Liver Disease'],
            colors=['#2ed573','#e94560'], autopct='%1.1f%%',
            startangle=90, textprops={'color':'white', 'fontsize':11}
        )
        for at in autotexts: at.set_color('white')
        ax.set_title("Target Distribution", color='white', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        # Feature importance from RF
        rf_model = trained_models["Random Forest"]
        importances = rf_model.feature_importances_
        feat_names  = X.columns.tolist()
        sorted_idx  = np.argsort(importances)

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1a1a2e')
        bars = ax.barh(
            [feat_names[i] for i in sorted_idx],
            importances[sorted_idx],
            color='#0f3460', edgecolor='#e94560', linewidth=0.5
        )
        bars[-1].set_color('#e94560')
        ax.set_title("Feature Importance (RF)", color='white', fontweight='bold')
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values(): spine.set_color('#0f3460')
        ax.xaxis.label.set_color('white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Feature distributions
    st.markdown("### Feature Distributions by Class")
    feature_sel = st.selectbox("Select Feature", X.columns.tolist())

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1a1a2e')
    for val, label, color in [(0,'No Disease','#2ed573'),(1,'Liver Disease','#e94560')]:
        subset = df[df['liver_disease']==val][feature_sel]
        ax.hist(subset, bins=30, alpha=0.6, label=label, color=color, edgecolor='none')
    ax.set_xlabel(feature_sel, color='white')
    ax.set_ylabel("Count", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    for spine in ax.spines.values(): spine.set_color('#0f3460')
    ax.set_title(f"Distribution of {feature_sel}", color='white', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Correlation heatmap
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0e1117')
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                mask=mask, annot_kws={"size": 8},
                linewidths=0.5, linecolor='#0e1117')
    ax.tick_params(colors='white', labelsize=8)
    ax.set_title("Feature Correlation Matrix", color='white', fontweight='bold')
    fig.patch.set_facecolor('#0e1117')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    ### 🫀 About This Project

    This web application predicts the likelihood of liver disease based on 9 clinical parameters
    using multiple machine learning models trained and compared from scratch.

    #### 🔬 Models Used
    - **Logistic Regression** — Baseline linear classifier
    - **Random Forest** — Bagging ensemble, 100 estimators
    - **Gradient Boosting** — Sequential boosting ensemble
    - **XGBoost** — Optimized gradient boosting
    - **AdaBoost** — Adaptive boosting
    - **Decision Tree** — Interpretable single tree
    - **KNN** — Distance-based classifier
    - **SVM (RBF)** — Support Vector Machine with RBF kernel
    - **Voting Classifier** ⭐ — Soft voting ensemble (LR + RF + XGBoost)

    #### ⚙️ Pipeline
    1. Data loading & EDA
    2. Train-test split (80/20, random_state=20)
    3. StandardScaler normalization
    4. SMOTE oversampling for class balance
    5. Model training & cross-validation
    6. ROC-AUC, Confusion Matrix, Classification Report evaluation

    #### 📊 Dataset
    - **2,000 patient records**, 9 liver function test features
    - Target: `liver_disease` (0 = No Disease, 1 = Disease)

    #### 👩‍💻 Developer
    **Akula Dhanalaxmi** | B.Tech CSE, SR University  
    [GitHub](https://github.com/AkulaDhanalaxmi) · [LinkedIn](https://www.linkedin.com/in/akula-dhanalaxmi-03b24934b) · [LeetCode](https://leetcode.com/u/Dhanalaxmiii/)

    ---
    > ⚠️ **Disclaimer**: This tool is for educational purposes only. Not a substitute for professional medical diagnosis.
    """)
