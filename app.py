
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tempfile
import mediapipe as mp
from math import atan2, degrees

st.set_page_config(page_title="StrideMatch Demo", layout="wide")

st.title("StrideMatch – Mini démo BlazePose + Reco chaussure")

with st.sidebar:
    st.header("Paramètres utilisateur")
    weight = st.number_input("Poids (kg)", min_value=30, max_value=200, value=75, step=1)
    weekly_km = st.number_input("Km hebdo", min_value=0, max_value=300, value=30, step=1)
    surface = st.selectbox("Surface principale", ["road", "trail"])
    pronation = st.selectbox("Pronation déclarée", ["neutral", "overpronation", "underpronation"])
    preference = st.selectbox("Préférence", ["confort", "stabilité", "réactivité"])
    catalog_file = st.file_uploader("Catalogue chaussures (CSV)", type=["csv"])
    st.caption("Colonnes requises : brand, model, stability, cushioning, terrain, stack_mm, drop_mm, weight_g, notes, link")

tab1, tab2 = st.tabs(["Analyse vidéo", "Recommandation"])

mp_pose = mp.solutions.pose

def angle(a, b, c):
    # a, b, c are (x,y) in pixels
    ang = degrees(atan2(c[1]-b[1], c[0]-b[0]) - atan2(a[1]-b[1], a[0]-b[0]))
    return abs((ang + 360) % 360)

with tab1:
    st.subheader("Uploader une courte vidéo (profil, 5–10 s, 720p)")
    up = st.file_uploader("Vidéo .mp4 .mov", type=["mp4","mov","avi"], accept_multiple_files=False, key="vid")
    run_btn = st.button("Analyser la vidéo")
    results_container = st.empty()

    if run_btn and up is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(up.read())
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_every = max(1, int(fps // 5))  # ~5 Hz

        knee_flex = []
        hip_flex = []
        ankle_dorsi = []
        trunk_lean = []

        with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if i % sample_every != 0:
                    i += 1
                    continue
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    # Indices MediaPipe
                    # Hip: 24/23, Knee: 26/25, Ankle: 28/27, Heel: 30/29, Toe: 32/31
                    # Choix côté droit pour cohérence caméra profil; si visibilité faible, tenter gauche
                    def get(p): 
                        return (int(lm[p].x*w), int(lm[p].y*h), lm[p].visibility)

                    right = {"hip":get(24), "knee":get(26), "ankle":get(28), "heel":get(30), "toe":get(32), "shoulder":get(12)}
                    left  = {"hip":get(23), "knee":get(25), "ankle":get(27), "heel":get(29), "toe":get(31), "shoulder":get(11)}
                    side = right if right["hip"][2] > left["hip"][2] else left

                    A = (side["hip"][0], side["hip"][1])
                    B = (side["knee"][0], side["knee"][1])
                    C = (side["ankle"][0], side["ankle"][1])
                    S = (side["shoulder"][0], side["shoulder"][1])
                    T = (side["toe"][0], side["toe"][1])
                    H = (side["heel"][0], side["heel"][1])

                    knee_ang = angle(A,B,C)        # knee flexion approx
                    hip_ang  = angle(S,A,B)        # hip flex/ext approx
                    ankle_ang = angle(B,C,T)       # ankle dorsiflex/plantar approx
                    trunk_ang = angle((A[0],A[1]-50), A, S)  # trunk lean vs vertical proxy

                    knee_flex.append(knee_ang)
                    hip_flex.append(hip_ang)
                    ankle_dorsi.append(ankle_ang)
                    trunk_lean.append(trunk_ang)
                i += 1
            cap.release()

        def summarize(arr):
            arr = np.array(arr)
            return {"mean": float(np.nanmean(arr)) if len(arr) else np.nan,
                    "p10": float(np.nanpercentile(arr,10)) if len(arr) else np.nan,
                    "p90": float(np.nanpercentile(arr,90)) if len(arr) else np.nan,
                    "n": int(len(arr))}

        summary = {
            "knee_deg": summarize(knee_flex),
            "hip_deg": summarize(hip_flex),
            "ankle_deg": summarize(ankle_dorsi),
            "trunk_deg": summarize(trunk_lean),
            "frames_used": len(knee_flex)
        }
        st.success("Analyse terminée")
        st.json(summary)
        st.session_state["biomech_summary"] = summary
    else:
        st.info("Charge une vidéo puis clique Analyser.")

with tab2:
    st.subheader("Recommandation automatique")

    # Charger catalogue
    if catalog_file is not None:
        try:
            cat = pd.read_csv(catalog_file)
        except Exception as e:
            st.error(f"Erreur lecture CSV: {e}")
            cat = None
    else:
        cat = None
        st.warning("Charge un catalogue CSV pour activer la recommandation.")

    # Règles simples basées sur préférences et biomécanique résumée
    summary = st.session_state.get("biomech_summary", None)

    def score_row(row):
        s = 0.0
        # Terrain
        s += 1.0 if row.get("terrain","road")==surface else -1.0
        # Pronation -> stabilité
        if pronation == "overpronation":
            s += 1.5 if row.get("stability")=="stability" else -0.5
        elif pronation == "underpronation":
            s += 0.5 if row.get("stability")=="neutral" else -0.2
        else:
            s += 0.5 if row.get("stability")=="neutral" else 0.0
        # Préférence
        if preference == "confort":
            s += 1.0 if row.get("cushioning") in ["high","max"] else 0.0
        elif preference == "stabilité":
            s += 1.0 if row.get("stability")=="stability" else 0.0
        elif preference == "réactivité":
            s += 1.0 if row.get("cushioning") in ["medium","race"] and row.get("weight_g",999)>0 and row.get("weight_g")<285 else 0.0
        # Poids coureur
        if weight >= 85 and row.get("cushioning") in ["high","max"]:
            s += 0.5
        if weekly_km >= 60 and row.get("cushioning") in ["high","medium"]:
            s += 0.3
        # Biomécanique indicative
        if summary:
            knee_mean = summary["knee_deg"]["mean"]
            trunk_mean = summary["trunk_deg"]["mean"]
            # Trunk lean élevé -> favoriser stabilité
            if np.isfinite(trunk_mean) and trunk_mean > 20 and row.get("stability")=="stability":
                s += 0.4
            # Genou très fléchi -> plus d'amorti
            if np.isfinite(knee_mean) and knee_mean > 170 and row.get("cushioning") in ["high","max"]:
                s += 0.3
        return s

    if cat is not None:
        sc = cat.apply(score_row, axis=1)
        cat_scored = cat.copy()
        cat_scored["score"] = sc
        recs = cat_scored.sort_values("score", ascending=False).head(3)

        st.write("### Top 3")
        for _, r in recs.iterrows():
            st.markdown(f"**{r['brand']} {r['model']}** — score {r['score']:.2f}")
            st.markdown(f"- Stabilité: {r['stability']}  |  Amorti: {r['cushioning']}  |  Terrain: {r['terrain']}  |  Drop: {r['drop_mm']} mm  |  Stack: {r['stack_mm']} mm")
            st.markdown(f"- Notes: {r.get('notes','')}  |  [Fiche]({r.get('link','#')})")
            st.markdown("---")

        with st.expander("Voir le catalogue scoré"):
            st.dataframe(cat_scored.sort_values("score", ascending=False), use_container_width=True)

        st.caption("Règles démo. Remplace par ton propre modèle plus tard.")
    else:
        st.stop()

st.markdown("---")
st.write("Cette démo utilise MediaPipe BlazePose pour estimer des angles simples, puis applique un moteur de règles pour proposer des chaussures. Pour production, remplace les règles par un modèle ML ou un appel API vers un LLM avec sources (RunRepeat, RunningShoesGuru).")
