# Zurich Apartment Price Predictor

## Project Overview
This application predicts apartment prices in Zurich and surrounding areas based on various features including:
- Number of rooms
- Area (m²)
- Location (town)
- Building age (the unique feature added to improve the model)

## Features
The model uses several features to predict apartment prices:
- Basic apartment characteristics (rooms, area)
- Socioeconomic factors (population, employment rate, tax income)
- Luxury indicators (from apartment descriptions)
- Building age categories (from "New" to "Historic")

## Building Age Feature
The unique feature added to this model is the building age, which categorizes apartments into:
- New (0-10 years old)
- Modern (11-30 years old)
- Established (31-50 years old)
- Older (51-70 years old)
- Historic (71+ years old)

Building age significantly impacts apartment prices, making it a valuable addition to the model.

## How to Use
1. Select the number of rooms
2. Enter the apartment area in square meters
3. Choose the town/location
4. Select the building age category
5. Click submit to get the predicted price

## Development
This application was developed using:
- Python
- Scikit-learn (RandomForestRegressor)
- Gradio for the web interface
- Deployed on Hugging Face Spaces

## Model Performance
Adding the building age feature improved the model's RMSE (Root Mean Square Error) from the baseline.