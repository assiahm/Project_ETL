import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape
import json

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

gdf_bus, gdf_travaux = charger_donnees()

st.title("🚌 Carte interactive des lignes de bus et travaux - Rennes Métropole")

# Dates min/max pour le filtre
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
    "Filtrer les travaux par période",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

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

# Filtrer sur la colonne niv_perturbation
if "niv_perturbation" in gdf_travaux_filtre.columns:
    niv_options = gdf_travaux_filtre["niv_perturbation"].dropna().unique().tolist()
    niv_options.sort()
    niv_selected = st.multiselect(
        "Filtrer les travaux par niveau de perturbation",
        options=niv_options,
        default=niv_options
    )
    if niv_selected:
        gdf_travaux_filtre = gdf_travaux_filtre[gdf_travaux_filtre["niv_perturbation"].isin(niv_selected)]

# Création de la carte
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
