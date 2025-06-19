import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt

# ---------------------- TITRE ----------------------
st.set_page_config(page_title="Prédiction STAR", layout="wide")
st.title("🚍 Prédiction de la fréquentation du réseau STAR (XGBoost)")
st.markdown("Utilisez ce modèle pour prédire la fréquentation journalière selon les caractéristiques d'une ligne.")

# ---------------------- FONCTION DE CHARGEMENT ----------------------
@st.cache_data
def load_data():
    try:
        engine = create_engine("postgresql+psycopg2://postgres:Amyas01062015%40@localhost:5432/olist_dwh")
        df = pd.read_sql_query("""
            SELECT annee, mois, date, "jourSemaine", semaine, "identifiantLigne", "nomCourtLigne", 
                   "typeLigne", "categorieLigne", "Frequentation", nom_vacances
            FROM "DWH_Projet_etudes".dfp_frequentations
        """, engine)
        return df
    except Exception as e:
        st.error(f"Erreur de connexion ou chargement des données : {e}")
        return pd.DataFrame()

df = load_data()

# Vérification
if df.empty:
    st.stop()

# ---------------------- PRÉTRAITEMENT ----------------------



# Encodage catégoriel
df['jourSemaine'] = df['jourSemaine'].astype('category').cat.codes
df['mois'] = df['mois'].astype('category').cat.codes
df['nomCourtLigne'] = df['nomCourtLigne'].astype('category').cat.codes
df['typeLigne'] = df['typeLigne'].astype('category').cat.codes
df['categorieLigne'] = df['categorieLigne'].astype('category').cat.codes
df['nom_vacances'] = df['nom_vacances'].fillna('Aucune').astype('category').cat.codes

# Création de la variable "type_jour"
def get_type_jour(row):
    if row['nom_vacances'] != 0:
        return 'vacances'
    elif row['jourSemaine'] in [5, 6]:
        return 'weekend'
    else:
        return 'travail'

df['type_jour'] = df.apply(get_type_jour, axis=1).astype('category').cat.codes

# ---------------------- FEATURES ET CIBLE ----------------------
features = [
    'annee', 'mois', 'jourSemaine', 'semaine', 'identifiantLigne',
    'nomCourtLigne', 'typeLigne', 'categorieLigne', 'nom_vacances', 'type_jour'
]
X = df[features]
y = df['Frequentation']

# Nettoyage
mask = ~y.isna()
X = X[mask]
y = y[mask]

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------- SLIDER TEST SIZE ----------------------
test_size = st.slider("🧪 Taille du jeu de test (%)", min_value=10, max_value=50, value=20)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size / 100, random_state=42
)

# ---------------------- PARAMÈTRES DU MODÈLE ----------------------
with st.expander("🛠️ Paramètres du modèle XGBoost"):
    n_estimators = st.selectbox("🌲 n_estimators", [100, 200, 300], index=0)
    max_depth = st.selectbox("🌳 max_depth", [6, 10, 15], index=0)
    learning_rate = st.selectbox("📉 learning_rate", [0.01, 0.05, 0.1], index=2)
    subsample = st.selectbox("🎯 subsample", [0.8, 1.0], index=1)

# ---------------------- ENTRAÎNEMENT ----------------------
# Définir last_date avant toute utilisation
last_date = pd.to_datetime(df['date']).max()

future_start = st.date_input(
    "📅 Date de début de prédiction (future)",
    value=last_date + pd.Timedelta(days=1),
    min_value=last_date + pd.Timedelta(days=1)
)
future_end = st.date_input(
    "📅 Date de fin de prédiction (future)",
    value=last_date + pd.Timedelta(days=61),
    min_value=future_start
)
if st.button("🚀 Entraîner le modèle"):
    with st.spinner("Optimisation en cours..."):
        model = GridSearchCV(
            XGBRegressor(random_state=42),
            param_grid={
                'n_estimators': [n_estimators],
                'max_depth': [max_depth],
                'learning_rate': [learning_rate],
                'subsample': [subsample]
            },
            scoring='neg_mean_absolute_error',
            cv=3,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        best_model = model.best_estimator_

        # Prédiction
        y_pred = best_model.predict(X_test)

        # Évaluation
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Filtres interactifs
        st.success("✅ Modèle entraîné avec succès")
        st.write("**Meilleurs paramètres XGBoost :**", model.best_params_)

        # Générer les dates futures sélectionnées
        date_range = pd.date_range(start=future_start, end=future_end, freq='D')

        # Générer un DataFrame pour les futures dates à prédire
        lignes_uniques = df['nomCourtLigne'].astype(str).unique()
        ligne_selection = st.multiselect(
            "🚌 Sélectionnez la/les ligne(s) à prédire",
            options=lignes_uniques,
            default=lignes_uniques.tolist()
        )

        # Pour chaque combinaison de date et ligne sélectionnée, générer les features nécessaires
        future_rows = []
        for date in date_range:
            for ligne in ligne_selection:
                # Récupérer les valeurs typiques pour la ligne (par exemple, le mode, ou la première occurrence)
                ligne_info = df[df['nomCourtLigne'].astype(str) == ligne].iloc[0]
                annee = date.year
                mois = date.month
                jourSemaine = date.weekday()
                semaine = date.isocalendar()[1]
                identifiantLigne = ligne_info['identifiantLigne']
                nomCourtLigne = ligne_info['nomCourtLigne']
                typeLigne = ligne_info['typeLigne']
                categorieLigne = ligne_info['categorieLigne']
                # On suppose pas de vacances pour le futur
                nom_vacances = 0
                # Type de jour
                if nom_vacances != 0:
                    type_jour = 2  # vacances
                elif jourSemaine in [5, 6]:
                    type_jour = 1  # weekend
                else:
                    type_jour = 0  # travail

                future_rows.append({
                    'annee': annee,
                    'mois': mois,
                    'jourSemaine': jourSemaine,
                    'semaine': semaine,
                    'identifiantLigne': identifiantLigne,
                    'nomCourtLigne': nomCourtLigne,
                    'typeLigne': typeLigne,
                    'categorieLigne': categorieLigne,
                    'nom_vacances': nom_vacances,
                    'type_jour': type_jour,
                    'date': date,
                    'nomCourtLigne_str': ligne
                })

        if not future_rows:
            st.warning("Aucune date ou ligne sélectionnée pour la prédiction future.")
        else:
            df_future = pd.DataFrame(future_rows)
            X_future = df_future[features]
            X_future_scaled = scaler.transform(X_future)
            y_pred_future = best_model.predict(X_future_scaled)
            df_future['Prédiction fréquentation'] = y_pred_future

            st.dataframe(df_future[['date', 'nomCourtLigne_str', 'Prédiction fréquentation']].rename(
            columns={'nomCourtLigne_str': 'nomCourtLigne'}
            ))

        # Filtre sur les lignes
        lignes_uniques = df['nomCourtLigne'].astype(str).unique()
        ligne_selection = st.multiselect(
            "🚌 Sélectionnez la/les ligne(s)",
            options=lignes_uniques,
            default=lignes_uniques.tolist()
        )

        # Filtrage du DataFrame d'origine selon la sélection utilisateur
        mask_dates = (pd.to_datetime(df['date']) >= pd.to_datetime(date_range[0])) & (pd.to_datetime(df['date']) <= pd.to_datetime(date_range[1]))
        mask_lignes = df['nomCourtLigne'].astype(str).isin(ligne_selection)
        df_filtre = df[mask_dates & mask_lignes]

        if df_filtre.empty:
            st.warning("Aucune donnée pour la sélection.")
        else:
            X_filtre = df_filtre[features]
            X_filtre_scaled = scaler.transform(X_filtre)
            # Entraînement du modèle en amont (hors interaction utilisateur)
            model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=42
            )
            model.fit(X_train, y_train)
            y_pred_filtre = model.predict(X_filtre_scaled)
            df_filtre_result = df_filtre.copy()
            df_filtre_result['Prédiction fréquentation'] = y_pred_filtre

            st.dataframe(df_filtre_result[['date', 'nomCourtLigne', 'Frequentation', 'Prédiction fréquentation']])

        # Affichage des scores globaux
        col1, col2, col3 = st.columns(3)
        col1.metric("📉 MSE", f"{mse:.2f}")
        col2.metric("📉 MAE", f"{mae:.2f}")
        col3.metric("📈 R²", f"{r2:.3f}")

        # Importance
        st.subheader("📊 Importance des variables")
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_importance(best_model, importance_type='weight', max_num_features=10, ax=ax)
        st.pyplot(fig)
