# StrideMatch – Mini Démo IA + BlazePose

## Description
Prototype Streamlit illustrant le concept StrideMatch :
- Analyse vidéo de la foulée avec **MediaPipe BlazePose**.
- Extraction d'angles clés (hanche, genou, cheville, tronc).
- Recommandation automatique de chaussures via règles simples et catalogue CSV.

## Fichiers
- `app.py` : code principal Streamlit.
- `stridematch_sample_catalog.csv` : exemple de catalogue de chaussures.

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Utilisation
1. Ouvrez l'interface Streamlit.
2. Chargez une vidéo de profil (5–10 s).
3. Chargez le CSV du catalogue.
4. Obtenez les recommandations automatiques.

## Personnalisation
- Modifiez le CSV pour ajouter vos modèles.
- Ajustez les règles dans `score_row()`.
- Pour un moteur IA avancé, ajoutez un appel API GPT et intégrez des données externes (RunRepeat, RunningShoesGuru).
