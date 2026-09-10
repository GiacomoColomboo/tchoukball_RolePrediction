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
# Nel caricamento modelli (@st.cache_resource)
@st.cache_resource
def load_all_models(advanced_data=True, sex=0):
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

    if advanced_data:
        pipeline = joblib.load(os.path.join(models_dir, 'trained_pipeline_advanced.joblib'))
        gmm_mixed = joblib.load(os.path.join(models_dir, 'gmm_advanced_archetypes.joblib'))
        gmm = joblib.load(os.path.join(models_dir, f'gmm_advanced_{"m" if sex == 0 else "w"}_archetypes.joblib'))
        weights = joblib.load(os.path.join(models_dir, 'weights_advanced.npy'))
    else:
        pipeline = joblib.load(os.path.join(models_dir, 'trained_pipeline.joblib'))
        gmm_mixed = joblib.load(os.path.join(models_dir, 'gmm_archetypes.joblib'))
        gmm = joblib.load(os.path.join(models_dir, f'gmm_{"m" if sex == 0 else "w"}_archetypes.joblib'))
        weights = joblib.load(os.path.join(models_dir, 'weights.npy'))

    return pipeline, gmm_mixed, gmm, weights

# Load and display the user interface on the first page (page_1.py)
def page_1():
    #Input statistiche giocatore
    st.header("📋 Box-Score Partita")

    sex_choice = st.selectbox("Genere / Categoria", options=["Maschile (0)", "Femminile (1)"])
    sex = 1 if "Femminile" in sex_choice else 0
    scored = st.number_input("Punti Segnati (SCORED)", 0, 100, 0)
    defence = st.number_input("Difese (DEFENCE)", 0, 100, 0)
    caught = st.number_input("Prese (CAUGHT)", 0, 100, 0)
    dropped = st.number_input("Palle Cadute (DROPPED)", 0, 100, 0)
    given_point = st.number_input("Punti Concessi (GIVEN_POINT)", 0, 100, 0)
    foul = st.number_input("Falli (FOUL)", 0, 100, 0)
    shot_pct = st.number_input("Percentuale Realizzativa (%SHOT_SCORED)", 0.0, 100.0, 0.0, step=0.1)
    advanced_data = st.checkbox("Mostra Metriche Avanzate", value=True)

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

    button = st.button("Predict Role & Analyze Player")
    if button:
        st.write("Navigating to Page2...")
        page_2(input_df, advanced_data, sex)

def page_2(input_df, advanced_data, sex):
    pipeline, gmm_mixed, gmm, weights = load_all_models(advanced_data=advanced_data, sex=sex)
    tab1, tab2 = st.tabs(["🎯 Predizione & XAI", "🕸️ Profilo Radar"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        
        # Inferenza
        prob_wing = pipeline.predict_proba(input_df)[0][1]
        role = "Wing" if prob_wing >= 0.5 else "Pivot"
        y = np.array([1 if role == "Wing" else 0])
        
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

        button = st.button("Insert new Player Data")
        if button:
            st.write("Navigating to Page1...")
            page_1()

    with tab2:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🕸️ Silhouette del Giocatore")

            # Radar plot del giocatore
            categories = list(input_df.columns)
            values = input_df.iloc[0].values.tolist()

            fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', name='Giocatore'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🎯 Profilazione Archetipica")
            
            X_trans_adv = pipeline.named_steps['trans'].transform(input_df)

            # 1. Inferenza cluster Mixed
            X_weighted_mixed = np.asarray(X_trans_adv) * np.sqrt(weights)
            cluster_mixed_id = gmm_mixed.predict(X_weighted_mixed)[0]
            
            # 2. Inferenza cluster Singola Categoria (Men / Women)
            X_trans_nosex = np.delete(np.asarray(X_trans_adv), -1, axis=1)
            X_weighted_gender = X_trans_nosex * np.sqrt(weights[:-1])
            
            if sex == 0:
                cluster_gender_id = gmm.predict(X_weighted_gender)[0]
                cat_name = "Maschile"
            else:
                cluster_gender_id = gmm.predict(X_weighted_gender)[0]
                cat_name = "Femminile" 
                
            # Testo descrittivo richiesto
            st.markdown(f"""
            ### 🌐 Inquadramento Mixed
            Per una competizione **Mixed**, questo giocatore fa parte dell'archetipo **"{(f'Archetipo {cluster_mixed_id}')}"**, con uno stile di gioco comparabile a.
            
            
            ---
            
            ### 🏆 Inquadramento Categoria {cat_name}
            Se si considera invece una competizione **non mixed** ({cat_name.lower()}), il profilo corrisponde all'archetipo **"{(f'Archetipo {cluster_gender_id}')}"**, simile a.
            
            """)

        button = st.button("Insert new Player Data")
        if button:
            st.write("Navigating to Page1...")
            page_1()

# Use Streamlit's built-in 'next' function to navigate between the pages
if __name__ == "__main__":
    page_1()