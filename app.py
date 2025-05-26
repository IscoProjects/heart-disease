# app.py
from flask import Flask, render_template_string, request
import mlflow.pyfunc
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar el modelo desde el registro de MLflow
mlflow.set_tracking_uri("http://localhost:9090")
model = mlflow.sklearn.load_model("models:/Heart_desease_prediction/2")

app = Flask(__name__)


# Ruta de inicio descriptiva
@app.route("/home")
def home():
    return "<h2>Bienvenido al sistema de predicción de enfermedades cardíacas</h2><p>Ir a <a href='/predict'>/predict</a> para realizar una predicción.</p>"


# Formulario para ingresar los datos
@app.route("/predict", methods=["GET"])
def predict_form():
    return render_template_string("""
    <h2>Formulario de Predicción de Enfermedad Cardíaca</h2>
    <form method="post" action="/result">
        <label>Edad:</label><input type="number" name="age" required><br><br>
        
        <label>Sexo:</label>
        <select name="sex">
            <option value="1">Hombre</option>
            <option value="0">Mujer</option>
        </select><br><br>
        
        <label>Tipo de dolor torácico (cp):</label>
        <select name="cp">
            <option value="0">Tipo 0</option>
            <option value="1">Tipo 1</option>
            <option value="2">Tipo 2</option>
            <option value="3">Tipo 3</option>
        </select><br><br>

        <label>Presión en reposo (trestbps):</label><input type="number" name="trestbps" required><br><br>
        <label>Colesterol (chol):</label><input type="number" name="chol" required><br><br>
        
        <label>Glucemia en ayunas > 120 mg/dl (fbs):</label>
        <select name="fbs">
            <option value="1">Sí</option>
            <option value="0">No</option>
        </select><br><br>

        <label>Resultado ECG en reposo (restecg):</label>
        <select name="restecg">
            <option value="0">Normal</option>
            <option value="1">Anormalidad ST-T</option>
            <option value="2">Hipertrofia ventricular izquierda</option>
        </select><br><br>

        <label>Frecuencia cardíaca máxima (thalach):</label><input type="number" name="thalach" required><br><br>
        <label>Angina inducida por ejercicio (exang):</label>
        <select name="exang">
            <option value="1">Sí</option>
            <option value="0">No</option>
        </select><br><br>

        <label>Oldpeak:</label><input type="number" step="0.1" name="oldpeak" required><br><br>
        <label>Pendiente del ST (slope):</label>
        <select name="slope">
            <option value="0">Tipo 0</option>
            <option value="1">Tipo 1</option>
            <option value="2">Tipo 2</option>
        </select><br><br>
        <label>Número de vasos (ca):</label>
        <select name="ca">
            <option value="0">0</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
        </select><br><br>
        <label>Thal:</label>
        <select name="thal">
            <option value="0">Normal</option>
            <option value="1">Defecto fijo</option>
            <option value="2">Defecto reversible</option>
        </select><br><br>

        <h4>Estilo de vida</h4>

        <label>Actividad física:</label>
        <select name="physical_activity">
            <option value="baja">Baja</option>
            <option value="moderada">Moderada</option>
            <option value="alta">Alta</option>
        </select><br><br>

        <label>¿Fuma?:</label>
        <select name="smoking_status">
            <option value="no fuma">No</option>
            <option value="fuma">Sí</option>
        </select><br><br>

        <label>Consumo de alcohol:</label>
        <select name="alcohol_intake">
            <option value="bajo">Bajo</option>
            <option value="medio">Medio</option>
            <option value="alto">Alto</option>
        </select><br><br>

        <label>Calidad de la dieta:</label>
        <select name="diet_quality">
            <option value="pobre">Pobre</option>
            <option value="regular">Regular</option>
            <option value="buena">Buena</option>
        </select><br><br>

        <input type="submit" value="Predecir">
    </form>
    """)


# Ruta para procesar predicción
@app.route("/result", methods=["POST"])
def result():
    data = request.form

    # Variables clínicas
    features = {
        'age': int(data['age']),
        'sex': int(data['sex']),
        'cp': int(data['cp']),
        'trestbps': int(data['trestbps']),
        'chol': int(data['chol']),
        'fbs': int(data['fbs']),
        'restecg': int(data['restecg']),
        'thalach': int(data['thalach']),
        'exang': int(data['exang']),
        'oldpeak': float(data['oldpeak']),
        'slope': int(data['slope']),
        'ca': int(data['ca']),
        'thal': int(data['thal']),
    }

    # Variables codificadas de estilo de vida
    features.update({
        'physical_activity_baja': data['physical_activity'] == 'baja',
        'physical_activity_moderada': data['physical_activity'] == 'moderada',
        'smoking_status_no fuma': data['smoking_status'] == 'no fuma',
        'alcohol_intake_bajo': data['alcohol_intake'] == 'bajo',
        'alcohol_intake_medio': data['alcohol_intake'] == 'medio',
        'diet_quality_pobre': data['diet_quality'] == 'pobre',
        'diet_quality_regular': data['diet_quality'] == 'regular',
    })

    # Asegurar el orden de columnas como el del modelo entrenado
    ordered_cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'physical_activity_baja',
        'physical_activity_moderada', 'smoking_status_no fuma',
        'alcohol_intake_bajo', 'alcohol_intake_medio', 'diet_quality_pobre',
        'diet_quality_regular'
    ]

    input_df = pd.DataFrame([features])[ordered_cols]

    # Realizar predicción
    prediction = model.predict(input_df)[0]
    resultado = "Positivo (riesgo de enfermedad cardíaca)" if prediction == 1 else "Negativo (sin riesgo detectable)"

    return f"<h3>Resultado de la predicción: {resultado}</h3><p><a href='/predict'>Volver al formulario</a></p>"


if __name__ == "__main__":
    app.run(debug=True)
