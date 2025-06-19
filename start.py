import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape
import json
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor, plot_importance
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

def couleur_perturbation(niv):
    if niv == "Secteur à éviter":
        return "black"
    elif niv == "Impact limité":
        return "yellow"
    elif niv == "Circulation difficile":
        return "red"
    else:
        return "green"

def charger_donnees():
    df_bus = pd.read_csv("Parcours des lignes de bus du réseau STAR.csv", delimiter=";")
    df_bus = df_bus.dropna(subset=["Parcours"])
    df_bus["geometry"] = df_bus["Parcours"].apply(lambda x: shape(json.loads(x)))
    gdf_bus = gpd.GeoDataFrame(df_bus, geometry="geometry", crs="EPSG:4326")

    df_travaux = pd.read_csv("Travaux de voirie impactant la circulation automobile sur Rennes Métropole pour le jour courant.csv", delimiter=";")
    df_travaux = df_travaux.dropna(subset=["Geo Shape"])
    df_travaux["geometry"] = df_travaux["Geo Shape"].apply(lambda x: shape(json.loads(x)))
    df_travaux["date_deb"] = pd.to_datetime(df_travaux["date_deb"], errors="coerce")
    df_travaux["date_fin"] = pd.to_datetime(df_travaux["date_fin"], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(df_travaux["date_deb"]):
        df_travaux["date_deb"] = df_travaux["date_deb"].dt.tz_localize(None)
    if pd.api.types.is_datetime64tz_dtype(df_travaux["date_fin"]):
        df_travaux["date_fin"] = df_travaux["date_fin"].dt.tz_localize(None)
    gdf_travaux = gpd.GeoDataFrame(df_travaux, geometry="geometry", crs="EPSG:4326")

    return gdf_bus, gdf_travaux

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

# Onglets principaux
tabs = st.tabs(["🗺️ Carte interactive", "🚍 Prédiction fréquentation", "🛤️ Planificateur de trajet STAR"])

with tabs[0]:
    st.title("🚌 Carte interactive des lignes de bus et travaux - Rennes Métropole")
    gdf_bus, gdf_travaux = charger_donnees()

    # Filtres dans l'onglet Carte interactive
    with st.expander("Filtres Travaux", expanded=True):
        min_date = gdf_travaux["date_deb"].min()
        max_date = gdf_travaux["date_fin"].max()
        if hasattr(min_date, 'tzinfo') and min_date.tzinfo is not None:
            min_date = min_date.tz_localize(None)
        if hasattr(max_date, 'tzinfo') and max_date.tzinfo is not None:
            max_date = max_date.tz_localize(None)
        if hasattr(min_date, 'date'):
            min_date = min_date.date()
        if hasattr(max_date, 'date'):
            max_date = max_date.date()
        date_range = st.date_input(
            "Période",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        # Filtre niveau perturbation
        if "niv_perturbation" in gdf_travaux.columns:
            niv_options = gdf_travaux["niv_perturbation"].dropna().unique().tolist()
            niv_options.sort()
            niv_selected = st.multiselect(
                "Niveau de perturbation",
                options=niv_options,
                default=niv_options
            )
        else:
            niv_selected = []

    # Application des filtres
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start = pd.to_datetime(date_range[0])
        end = pd.to_datetime(date_range[1])
        gdf_travaux["date_deb"] = pd.to_datetime(gdf_travaux["date_deb"], errors="coerce")
        gdf_travaux["date_fin"] = pd.to_datetime(gdf_travaux["date_fin"], errors="coerce")
        if pd.api.types.is_datetime64tz_dtype(gdf_travaux["date_deb"]):
            gdf_travaux["date_deb"] = gdf_travaux["date_deb"].dt.tz_localize(None)
        if pd.api.types.is_datetime64tz_dtype(gdf_travaux["date_fin"]):
            gdf_travaux["date_fin"] = gdf_travaux["date_fin"].dt.tz_localize(None)
        gdf_travaux_filtre = gdf_travaux[
            (gdf_travaux["date_fin"].dt.date >= start.date()) & (gdf_travaux["date_deb"].dt.date <= end.date())
        ]
    else:
        gdf_travaux_filtre = gdf_travaux

    if niv_selected:
        gdf_travaux_filtre = gdf_travaux_filtre[gdf_travaux_filtre["niv_perturbation"].isin(niv_selected)]

    # Carte
    m = folium.Map(
        location=[48.117266, -1.677793],
        zoom_start=12,
        tiles="OpenStreetMap"
    )
    folium.GeoJson(
        gdf_bus,
        name="Lignes de bus",
        style_function=lambda _: {"color": "blue", "weight": 2, "opacity": 0.7},
        tooltip=folium.GeoJsonTooltip(fields=["Ligne (ID)"], aliases=["Ligne :"])
    ).add_to(m)

    gdf_travaux_filtre_serializable = gdf_travaux_filtre.drop(columns=["date_deb", "date_fin"], errors="ignore")
    if (
        "niv_perturbation" in gdf_travaux_filtre_serializable.columns
        and not gdf_travaux_filtre_serializable["niv_perturbation"].isnull().all()
    ):
        folium.GeoJson(
            gdf_travaux_filtre_serializable,
            name="Travaux",
            style_function=lambda feature: {
                "color": couleur_perturbation(feature["properties"].get("niv_perturbation")),
                "weight": 4,
                "opacity": 0.8
            },
            tooltip=folium.GeoJsonTooltip(fields=["niv_perturbation"])
        ).add_to(m)
    else:
        folium.GeoJson(
            gdf_travaux_filtre_serializable,
            name="Travaux",
            style_function=lambda _: {"color": "yellow", "weight": 4, "opacity": 0.8}
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=1100, height=600)

with tabs[1]:
    import matplotlib.pyplot as plt

    # ---------------------- TITRE ----------------------
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

with tabs[2]:
    st.title("🛤️ Planificateur de trajet STAR")
    st.markdown("Planifiez votre itinéraire sur le réseau STAR (métro) entre deux stations.")

    # Chargement des données depuis la base PostgreSQL
    try:
        engine = create_engine("postgresql+psycopg2://postgres:Amyas01062015%40@localhost:5432/olist_ods")
        stations = pd.read_sql_query(
            'SELECT "Geo Point", "Geo Shape", gml_id, objectid, ligne, ordre, nom, x_cc48, y_cc48, x_wgs84, y_wgs84 FROM "ODS_Projet_etudes".dfa',
            engine
        )
    except Exception as e:
        st.error(f"Erreur de connexion ou chargement des stations : {e}")
        st.stop()

    stations = stations.dropna(subset=['Geo Point'])

    # Création du graphe simple basé sur les lignes de métro
    G = nx.DiGraph()

    for _, row in stations.iterrows():
        lat, lon = map(float, row['Geo Point'].split(','))
        G.add_node(row['nom'], pos=(lat, lon), ligne=row['ligne'], ordre=row['ordre'])

    for ligne in stations['ligne'].dropna().unique():
        df_ligne = stations[stations['ligne'] == ligne].sort_values('ordre')
        for i in range(len(df_ligne) - 1):
            u = df_ligne.iloc[i]['nom']
            v = df_ligne.iloc[i + 1]['nom']
            if not G.has_edge(u, v):
                G.add_edge(u, v, poids=1, ligne=ligne)
            if not G.has_edge(v, u):
                G.add_edge(v, u, poids=1, ligne=ligne)

    # Sélection des stations de départ et d'arrivée
    all_stations = sorted(G.nodes)
    col1, col2 = st.columns(2)
    with col1:
        depart = st.selectbox("📍 Station de départ", all_stations, key="depart")
    with col2:
        destination = st.selectbox("🎯 Station d'arrivée", all_stations, key="destination")

    if depart != destination:
        try:
            chemins = list(nx.all_simple_paths(G, source=depart, target=destination, cutoff=10))
            st.subheader("🔀 Itinéraires possibles")
            st.write(f"{len(chemins)} itinéraire(s) possible(s) entre **{depart}** et **{destination}**")

            for i, chemin in enumerate(chemins[:5]):
                poids_total = sum(G[chemin[j]][chemin[j + 1]].get('poids', 1) for j in range(len(chemin) - 1))
                lignes_segment = [
                    f"{chemin[j]} → {chemin[j+1]} : Ligne {G[chemin[j]][chemin[j+1]].get('ligne', 'N/A').upper()}"
                    for j in range(len(chemin) - 1)
                ]
                st.markdown(f"**Itinéraire {i+1}** : {' → '.join(chemin)}")
                st.markdown("<br>".join(lignes_segment), unsafe_allow_html=True)
                st.text(f"⏱️ Temps estimé : {poids_total:.2f} min\n")

            # Carte interactive du premier itinéraire
            if chemins:
                chemin = chemins[0]
                lat_depart, lon_depart = G.nodes[depart]['pos']
                m = folium.Map(location=[lat_depart, lon_depart], zoom_start=13)

                for station in chemin:
                    lat, lon = G.nodes[station]['pos']
                    ligne = G.nodes[station].get('ligne', 'N/A')
                    couleur = 'red' if ligne == 'a' else 'green' if ligne == 'b' else 'gray'
                    folium.Marker(
                        location=[lat, lon],
                        popup=f"{station} (Ligne {ligne.upper()})",
                        icon=folium.Icon(color=couleur)
                    ).add_to(m)

                for j in range(len(chemin) - 1):
                    u, v = chemin[j], chemin[j + 1]
                    lat_u, lon_u = G.nodes[u]['pos']
                    lat_v, lon_v = G.nodes[v]['pos']
                    ligne = G[u][v].get('ligne', 'N/A')
                    couleur = 'red' if ligne == 'a' else 'green' if ligne == 'b' else 'gray'
                    folium.PolyLine(locations=[[lat_u, lon_u], [lat_v, lon_v]], color=couleur, weight=4).add_to(m)

                st.subheader("🗺️ Carte de l'itinéraire optimal")
                st_folium(m, width=1000, height=600)

        except nx.NetworkXNoPath:
            st.error("❌ Aucun chemin trouvé entre ces deux stations.")
    else:
        st.warning("⚠️ Veuillez sélectionner deux stations différentes.")
