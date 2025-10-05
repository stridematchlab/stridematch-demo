import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="StrideMatch Demo (Simulé)", layout="wide")

st.title("StrideMatch – Démo conceptuelle sans MediaPipe")

with st.sidebar:
    st.header("Profil utilisateur")
    weight = st.number_input("Poids (kg)", 40, 150, 75)
    weekly_km = st.number_input("Km hebdo", 0, 200, 30)
    surface = st.selectbox("Surface principale", ["road", "trail"])
    pronation = st.selectbox("Pronation", ["neutral", "overpronation", "underpronation"])
    preference = st.selectbox("Préférence", ["confort", "stabilité", "réactivité"])
    catalog_file = st.file_uploader("Catalogue chaussures (CSV)", type=["csv"])

tab1, tab2 = st.tabs(["Analyse biomécanique simulée", "Recommandation"])

# --- Simulation section ---
with tab1:
    st.subheader("Simulation des angles de course")
    st.write("En version Cloud, la détection vidéo MediaPipe est désactivée. "
             "Des valeurs biomécaniques réalistes sont simulées pour illustrer le concept.")
    np.random.seed(42)
    biomech = {
        "knee_deg": round(np.random.normal(165, 5), 1),
        "hip_deg": round(np.random.normal(150, 8), 1),
        "ankle_deg": round(np.random.normal(95, 4), 1),
        "trunk_deg": round(np.random.normal(15, 3), 1)
    }
    st.json(biomech)
    st.session_state["biomech_summary"] = biomech

# --- Recommendation logic ---
with tab2:
    st.subheader("Recommandation de chaussures")
    if catalog_file is None:
        st.warning("Charge un fichier CSV de catalogue pour continuer.")
        st.stop()

    cat = pd.read_csv(catalog_file)

    summary = st.session_state.get("biomech_summary", biomech)

    def score_row(row):
        s = 0
        if row.get("terrain") == surface:
            s += 1
        if pronation == "overpronation" and row.get("stability") == "stability":
            s += 1.5
        if pronation == "neutral" and row.get("stability") == "neutral":
            s += 1
        if preference == "confort" and row.get("cushioning") in ["high","max"]:
            s += 1
        if preference == "réactivité" and row.get("weight_g",999) < 280:
            s += 1
        if weight > 85 and row.get("cushioning") in ["high","max"]:
            s += 0.5
        if summary["trunk_deg"] > 20 and row.get("stability") == "stability":
            s += 0.3
        return s

    cat["score"] = cat.apply(score_row, axis=1)
    recs = cat.sort_values("score", ascending=False).head(3)

    st.write("### Top 3 recommandations")
    for _, r in recs.iterrows():
        st.markdown(f"**{r['brand']} {r['model']}** — score {r['score']:.2f}")
        st.markdown(f"- Stabilité: {r['stability']}  |  Amorti: {r['cushioning']}  |  Terrain: {r['terrain']}")
        st.markdown(f"- [Voir la fiche]({r['link']})")
        st.markdown("---")

st.caption("Version démonstration. Les valeurs biomécaniques sont simulées pour un usage cloud sans MediaPipe.")
