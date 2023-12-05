import joblib
from feature_extraction import feature_extraction

xgb_model = joblib.load('xgb_model.joblib')
dt_model = joblib.load('dt_model.joblib')
rf_model = joblib.load('rf_model.joblib')
knn_model = joblib.load('knn_model.joblib')

def predict_phishing(url, models):
   
   # Extract features from URL
   features = feature_extraction(url)

   # Run model predictions
   predictions = {}
   for model_name, model in models.items():
      pred = model.predict([features])[0]
      predictions[model_name] = pred

   return predictions

models = {
    'XGBoost': xgb_model,
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'KNN': knn_model
}

# Exemplu de utilizare
url_to_check = "https://www.mummyandmini.com/"
predictions = predict_phishing(url_to_check, models)

# Afiseaza rezultatele
for model_name, prediction in predictions.items():
    print(f"{model_name} Prediction for {url_to_check}: {prediction}")