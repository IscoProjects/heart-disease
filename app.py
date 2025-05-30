# app.py
import io
from flask import Flask, render_template_string, request, render_template
import mlflow.pyfunc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Importante para evitar GUI
import base64

app = Flask(__name__)

# Cargar el modelo desde el registro de MLflow
mlflow.set_tracking_uri("http://localhost:9090")
model = mlflow.sklearn.load_model("models:/Heart_desease_prediction/3")

app = Flask(__name__)


# Ruta de inicio descriptiva
@app.route("/home")
def home():
    return render_template("home.html")


# Formulario para ingresar los datos
@app.route("/predict", methods=["GET", "POST"])
def predict_form():
    if request.method == "POST":
        # Procesar los datos del formulario
        ...
    return render_template("predict.html")


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
        'physical_activity': data['physical_activity'] == 'true',
        'smoking_status': data['smoking_status'] == 'true',
        'alcohol_intake': data['alcohol_intake'] == 'true',
        'diet_quality': data['diet_quality'] == 'true'
    }

    # Asegurar el orden de columnas como el del modelo entrenado
    ordered_cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'physical_activity',
        'smoking_status', 'alcohol_intake', 'diet_quality'
    ]

    input_df = pd.DataFrame([features])[ordered_cols]

    # Realizar predicción
    prediction = model.predict(input_df)[0]
    # En tu vista Flask o en el código de prueba
    proba = model.predict_proba(input_df)[0][1]
    # En tu vista Flask o en el código de prueba
    print(f"Probabilidad estimada de enfermedad cardíaca: {proba:.2f}")

    resultado = "Positivo (riesgo de enfermedad cardíaca)" if prediction == 1 else "Negativo (sin riesgo detectable)"

    # Variables clínicas seleccionadas
    clinical_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    clinical_min_max = {
        'age': (29, 77),
        'trestbps': (94, 200),
        'chol': (126, 564),
        'thalach': (71, 202),
        'oldpeak': (0.0, 6.2)
    }
    clinical_values = [(features[v] - clinical_min_max[v][0]) /
                       (clinical_min_max[v][1] - clinical_min_max[v][0])
                       for v in clinical_vars]

    # Estilo de vida agrupado
    lifestyle_vars = ['Actividad física', 'Fumador', 'Alcohol', 'Dieta']
    lifestyle_values = [
        1 if features['physical_activity'] else 0,
        1 if features['smoking_status'] else 0,
        1 if features['alcohol_intake'] else 0,
        1 if features['diet_quality'] else 0
    ]

    radar_clinical = plot_radar(clinical_vars.copy(), clinical_values.copy(),
                                "Variables Clínicas")
    radar_lifestyle = plot_radar(lifestyle_vars.copy(),
                                 lifestyle_values.copy(), "Estilo de Vida")

    return render_template("result.html",
                           resultado=resultado,
                           probabilidad=proba,
                           radar_clinical=radar_clinical,
                           radar_lifestyle=radar_lifestyle)


def plot_radar(labels, values, title):
    labels += [labels[0]]
    values += [values[0]]
    angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1])
    ax.set_yticklabels([])
    ax.set_title(title, y=1.1)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


if __name__ == "__main__":
    app.run(debug=True)
