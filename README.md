# 🫀 Predicción de Enfermedades Cardíacas con Machine Learning

Este proyecto aplica técnicas de aprendizaje automático para predecir el riesgo de enfermedad cardíaca en pacientes, utilizando información clínica y factores de estilo de vida. El modelo final ha sido desplegado mediante una aplicación web desarrollada con Flask, y los experimentos se gestionan con MLflow.

---

## 📌 Objetivos del proyecto

- Analizar datos clínicos y de estilo de vida de pacientes.
- Entrenar y comparar modelos de machine learning.
- Seleccionar el mejor modelo para la predicción del riesgo cardíaco.
- Desplegar el modelo mediante una API con Flask.
- Gestionar los experimentos usando MLflow.

---

## 🧠 Tecnologías y librerías utilizadas

- **Python 3.11**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Scikit-learn** para preprocesamiento y modelos
- **MLflow** para seguimiento y publicación del modelo
- **Flask** para construir una API de predicción
- **SQLite3** para el dataset de estilo de vida

---

## 🗃️ Dataset utilizado

Se combinaron dos fuentes de datos:

1. **Heart Disease Dataset (CSV):** Información clínica de pacientes como edad, presión arterial, colesterol, etc.
2. **Lifestyle Heart Risk Dataset (SQLite):** Factores de estilo de vida como actividad física, tabaquismo, alcohol e índice de dieta.

Las variables de estilo de vida se integraron con los datos clínicos.

---

## 🧪 Comparación de modelos

Se entrenaron y evaluaron los siguientes modelos:

| Modelo              | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Random Forest       | 0.980    | 0.962     | 1.000  | 0.980    |
| SVM                 | 0.886    | 0.857     | 0.932  | 0.893    |
| Logistic Regression | 0.881    | 0.849     | 0.932  | 0.888    |
| Decision Tree       | 0.871    | 0.873     | 0.873  | 0.873    |
| KNN                 | 0.856    | 0.824     | 0.912  | 0.866    |

**Random Forest** fue seleccionado como el mejor modelo.

---

## 🚀 Despliegue del modelo

- El modelo entrenado se registró y guardó con **MLflow**.
- Se construyó una API REST con **Flask** que permite a los usuarios ingresar su información y obtener una predicción del riesgo cardíaco.
- La aplicación se ejecuta localmente y está lista para ser subida a un entorno de producción.

---

## ▶️ Cómo ejecutar el proyecto

1. Clonar el repositorio:

```bash
git clone https://github.com/tuusuario/nombre-del-repo.git
cd nombre-del-repo
```

2. Crear y activar un entorno virtual

```bash
python -m venv venv
source venv/bin/activate
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

4. Iniciar MLflow en otro terminal:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 9090
```

5. Ejecutar la aplicación Flask:

```bash
python app.py
```

6. Accede a la aplicación desde tu navegador:

```bash
http://localhost:5000/home
```
