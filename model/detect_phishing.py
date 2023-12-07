import joblib
from feature_extraction import feature_extraction
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

xgb_model = joblib.load('xgb_model.joblib')
dt_model = joblib.load('dt_model.joblib')
rf_model = joblib.load('rf_model.joblib')
knn_model = joblib.load('knn_model.joblib')

def predict_phishing(models):
   url_to_check = input("Introduceți URL-ul de verificat: ")

   features = feature_extraction(url_to_check)

   predictions = {}
   for model_name, model in models.items():
      pred = model.predict([features])[0]
      predictions[model_name] = pred

   return url_to_check, predictions

models = {
   'XGBoost': xgb_model,
   'Decision Tree': dt_model,
   'Random Forest': rf_model,
   'KNN': knn_model
}

url_to_check, predictions = predict_phishing(models)

for model_name, prediction in predictions.items():
   print(f"\n{model_name} Prediction: {prediction}")