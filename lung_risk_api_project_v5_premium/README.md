# API de predicción del nivel de riesgo asociado al cáncer de pulmón

Versión premium del prototipo web en Flask para clasificar el nivel de riesgo asociado al cáncer de pulmón mediante un modelo Extra Trees Classifier.

## Variables de entrada

- Tabaquismo (`Smoking`)
- Contaminación del aire (`Air Pollution`)
- Consumo de alcohol (`Alcohol use`)
- Obesidad (`Obesity`)
- Riesgo genético (`Genetic Risk`)

## Salida

La API devuelve una categoría de riesgo:

- `Low` = Bajo
- `Medium` = Medio
- `High` = Alto

También devuelve probabilidades por clase y un reporte orientativo con recomendaciones preventivas.

## Ejecución en Windows usando el Python de Spyder

```cmd
cd "C:\Users\atafv\OneDrive\Escritorio\lung_risk_api_project_v4_premium"
"C:\ProgramData\spyder-6\envs\spyder-runtime\python.exe" -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Luego abrir:

```text
http://127.0.0.1:5000/
```

## Advertencia

El resultado es exploratorio y de apoyo preventivo. No constituye diagnóstico clínico ni reemplaza la evaluación de un profesional de salud.
