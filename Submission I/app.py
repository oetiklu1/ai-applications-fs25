import gradio as gr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor



# ------------------------------
# Lade das richtige CSV mit dem neuen Feature
# ------------------------------
df = pd.read_csv("apartments_data_enriched_with_new_features.csv")

# Sicherstellen, dass building_age existiert
if "building_age" not in df.columns:
    raise ValueError("❌ Fehler: Das Feature 'building_age' fehlt im CSV!")

# ------------------------------
# town bleibt als String im Gradio-Dropdown
# ------------------------------
towns = sorted(df["town"].unique().tolist())  # Liste aller Städte für Gradio
# Mapping Stadtname → numerische Werte für das Modell
town_mapping = {town: idx for idx, town in enumerate(towns)}

# ------------------------------
# Baujahr-Kategorien für das Dropdown
# ------------------------------
age_categories = ["New (0-10)", "Modern (11-30)", "Established (31-50)", "Older (51-70)", "Historic (71+)"]
# Mapping von Kategorie zu numerischem Wert
age_category_mapping = {cat: idx for idx, cat in enumerate(age_categories)}

# ------------------------------
# Trainiere das Modell mit numerischen Daten
# ------------------------------
features = ['rooms', 'area', 'pop', 'pop_dens', 'frg_pct', 'emp', 'tax_income', 
            'room_per_m2', 'luxurious', 'temporary', 'furnished', 
            'building_age', 'building_age_cat_encoded']

# Stelle sicher, dass town numerisch verarbeitet wird
df["town_numeric"] = df["town"].map(town_mapping)

# Trainiere das Modell
import pickle  
# Load model from file 
model_filename = "apartment_price_model.pkl"
with open(model_filename, mode="rb") as f:  
    model = pickle.load(f) 




# ------------------------------
# Vorhersagefunktion für Gradio
# ------------------------------
def predict_price(rooms, area, town, building_age_category):
    print()
    if town not in town_mapping:
        return "Error: Town not found in dataset."


    # Durchschnittswerte für andere Features aus der Stadt berechnen
    town_data = df[df["town"] == town][['pop', 'pop_dens', 'frg_pct', 'emp', 'tax_income', 
                                       'room_per_m2', 'luxurious', 'temporary', 'furnished']].mean()

    # Berechne Gebäudealter aus der Kategorie (Mittelpunkt der Kategorie)
    building_age = 0
    if building_age_category == "New (0-10)":
        building_age = 5
    elif building_age_category == "Modern (11-30)":
        building_age = 20
    elif building_age_category == "Established (31-50)":
        building_age = 40
    elif building_age_category == "Older (51-70)":
        building_age = 60
    else:  # Historic
        building_age = 85

    building_age_cat_encoded = age_category_mapping[building_age_category]

    # Erstelle das Input-Array für das Modell
    input_data = np.array([[rooms, area, town_data["pop"], town_data["pop_dens"], 
                            town_data["frg_pct"], town_data["emp"], town_data["tax_income"], 
                            town_data["room_per_m2"], town_data["luxurious"], town_data["temporary"], 
                            town_data["furnished"], building_age, building_age_cat_encoded]])

    # Preisvorhersage
    prediction = model.predict(input_data)[0]
    return f"Predicted Apartment Price: {round(prediction, 2)} CHF"


# ------------------------------
# Gradio Interface erstellen (MIT GEBÄUDEALTER)
# ------------------------------
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Rooms", value=3.5),
        gr.Number(label="Area (m²)", value=67),
        gr.Dropdown(label="Town", choices=towns, value=" Zürich"),  # Städtenamen bleiben als Text
        gr.Dropdown(label="Building Age Category", choices=age_categories, value="Established (31-50)")
    ],
    outputs="text",
    title="Zurich Apartment Price Estimator",
    description="Enter the apartment details to get an estimated price, including the effect of building age."
)

# ------------------------------
# Starte die Gradio-App
# ------------------------------
interface.launch()