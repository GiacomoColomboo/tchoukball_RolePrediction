import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# Configurazione pagina
st.set_page_config(page_title="Tchoukball Scouting Suite", page_icon="🤾", layout="wide")

st.title("🤾 Tchoukball Player Analytics & Role Predictor")
st.markdown("Strumento di scouting predittivo e profilazione archetipi basato su Machine Learning ed Explainable AI.")

# Caricamento modelli
@st.cache_resource
def load_models(advanced_data=True):
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    if advanced_data:
        pipeline = joblib.load(os.path.join(models_dir, 'trained_pipeline_advanced.joblib'))
    else:
        pipeline = joblib.load(os.path.join(models_dir, 'trained_pipeline.joblib'))
    return pipeline

advanced_data = st.sidebar.checkbox("Mostra Metriche Avanzate", value=True)
pipeline = load_models(advanced_data=advanced_data)

# Sidebar: Input statistiche giocatore
st.sidebar.header("📋 Box-Score Partita")
sex_choice = st.sidebar.selectbox("Genere / Categoria", options=["Maschile (0)", "Femminile (1)"])
sex = 1 if "Femminile" in sex_choice else 0

scored = st.sidebar.number_input("Punti Segnati (SCORED)", 0, 100, 0)
defence = st.sidebar.number_input("Difese (DEFENCE)", 0, 100, 0)
caught = st.sidebar.number_input("Prese (CAUGHT)", 0, 100, 0)
dropped = st.sidebar.number_input("Palle Cadute (DROPPED)", 0, 100, 0)
given_point = st.sidebar.number_input("Punti Concessi (GIVEN_POINT)", 0, 100, 0)
foul = st.sidebar.number_input("Falli (FOUL)", 0, 100, 0)
shot_pct = st.sidebar.number_input("Percentuale Realizzativa (%SHOT_SCORED)", 0.0, 100.0, 0.0, step=0.1)
shot_pct_radar = shot_pct / 10.0

if advanced_data:
    # Calcolo metriche ingegnerizzate al volo
    input_df = pd.DataFrame([{
        'SCORED': scored, 'DEFENCE': defence, 'CAUGHT': caught, 'DROPPED': dropped,
        'GIVEN_POINT': given_point, 'FOUL': foul, '%SHOT_SCORED': shot_pct,
        'OFF_DEF_RATIO': scored / (defence + caught + 1.0),
        'CATCH_EFFICIENCY': caught / (caught + dropped + 1.0),
        'NET_POINTS': scored - given_point,
        'OFF_LOAD_SHARE': scored / (scored + defence + caught + given_point + foul + 1.0),
        'ERROR_PRONENESS': (foul + given_point) / (scored + defence + caught + 1.0),
        'SEX': sex
    }])
else:
    input_df = pd.DataFrame([{
        'SCORED': scored, 'DEFENCE': defence, 'CAUGHT': caught, 'DROPPED': dropped,
        'GIVEN_POINT': given_point, 'FOUL': foul, '%SHOT_SCORED': shot_pct,
        'SEX': sex
    }])

tab1, tab2 = st.tabs(["🎯 Predizione & XAI", "🕸️ Profilo Radar"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    # Inferenza
    prob_wing = pipeline.predict_proba(input_df)[0][1]
    role = "Wing" if prob_wing >= 0.5 else "Pivot"
    
    with col1:
        if role == "Wing":
            st.metric(label="Ruolo Predetto", value=role, delta=f"{prob_wing*100:.1f}% Confidenza Ala")
            st.progress(prob_wing)
        else:
            st.metric(label="Ruolo Predetto", value=role, delta=f"{(1-prob_wing)*100:.1f}% Confidenza Pivot")
            st.progress(1 - prob_wing)
        if advanced_data:
            st.write(f"**Indice Offensivo (OFF_DEF_RATIO):** `{input_df['OFF_DEF_RATIO'].values[0]:.2f}`")
            st.write(f"**Efficienza di Presa:** `{input_df['CATCH_EFFICIENCY'].values[0]*100:.1f}%`")
            st.write(f"**Punti Netti:** `{input_df['NET_POINTS'].values[0]:.0f}`")
        
    with col2:
        st.subheader("🔍 Spiegazione Decisionale (SHAP Waterfall)")
        X_trans = pipeline.named_steps['trans'].transform(input_df)
        feature_names = [f.split('__')[-1] for f in pipeline.named_steps['trans'].get_feature_names_out()]
        background = np.zeros((1, X_trans.shape[1]))
        explainer = shap.LinearExplainer(pipeline.named_steps['classifier'], background, feature_names=feature_names)
        shap_val = explainer(X_trans)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_val[0], show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab2:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🕸️ Silhouette del Giocatore")
        if advanced_data:
            categories = ['SCORED', 'DEFENCE', 'CAUGHT', 'DROPPED', '%SHOT_SCORED', 'OFF_DEF_RATIO', 'CATCH_EFFICIENCY', 'NET_POINTS']
            values = [scored, defence, caught, dropped, shot_pct_radar, input_df['OFF_DEF_RATIO'].values[0], input_df['CATCH_EFFICIENCY'].values[0], input_df['NET_POINTS'].values[0]]
        else:
            categories = ['SCORED', 'DEFENCE', 'CAUGHT', 'DROPPED', '%SHOT_SCORED']
            values = [scored, defence, caught, dropped, shot_pct_radar]

        fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', name='Giocatore'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Archetype del Giocatore")
        
