from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

app = Flask(__name__)

model = joblib.load(ARTIFACTS_DIR / "extra_trees_lung_risk_model.pkl")
scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
features = joblib.load(ARTIFACTS_DIR / "features.pkl")

RISK_TRANSLATION = {
    "Low": "Bajo",
    "Medium": "Medio",
    "High": "Alto",
}

VARIABLE_LABELS = {
    "Smoking": "Tabaquismo",
    "Air Pollution": "Contaminación del aire",
    "Alcohol use": "Consumo de alcohol",
    "Obesity": "Obesidad",
    "Genetic Risk": "Riesgo genético",
}


VARIABLE_IMAGES = {
    "Smoking": "img/smoking.svg",
    "Air Pollution": "img/pollution.svg",
    "Alcohol use": "img/alcohol.svg",
    "Obesity": "img/obesity.svg",
    "Genetic Risk": "img/genetic.svg",
}

REPORT_IMAGES = {
    "hero": "img/hero-lungs.svg",
    "summary": "img/report-summary.svg",
    "advice": "img/report-advice.svg",
    "warning": "img/report-warning.svg",
}
VALID_RANGES = {
    "Smoking": (1, 8),
    "Air Pollution": (1, 8),
    "Alcohol use": (1, 8),
    "Obesity": (1, 7),
    "Genetic Risk": (1, 7),
}

VARIABLE_OPTIONS = {
    "Smoking": [
        {"value": 1, "label": "Muy bajo o nulo", "example": "No fuma o exposición casi inexistente al humo de tabaco."},
        {"value": 2, "label": "Muy bajo", "example": "Exposición ocasional; no fuma de forma habitual."},
        {"value": 3, "label": "Bajo", "example": "Fuma muy poco o se expone al humo algunos días."},
        {"value": 4, "label": "Bajo-regular", "example": "Fuma pocos cigarrillos al día o tiene exposición semanal."},
        {"value": 5, "label": "Moderado", "example": "Fuma con frecuencia, aproximadamente 4 a 9 cigarrillos/día."},
        {"value": 6, "label": "Alto", "example": "Fuma diariamente, aproximadamente 10 a 19 cigarrillos/día."},
        {"value": 7, "label": "Muy alto", "example": "Fuma 20 o más cigarrillos/día o tiene exposición intensa diaria."},
        {"value": 8, "label": "Extremo", "example": "Consumo muy intenso o exposición constante al humo de tabaco."},
    ],
    "Air Pollution": [
        {"value": 1, "label": "Muy baja", "example": "Zona con aire limpio, poco tránsito o ambiente rural."},
        {"value": 2, "label": "Baja", "example": "Exposición ligera a tránsito o humo ambiental."},
        {"value": 3, "label": "Leve", "example": "Zona residencial con tránsito bajo a moderado."},
        {"value": 4, "label": "Regular", "example": "Exposición frecuente a tráfico vehicular o polvo urbano."},
        {"value": 5, "label": "Moderada", "example": "Vive o trabaja cerca de avenidas transitadas varias horas al día."},
        {"value": 6, "label": "Alta", "example": "Exposición diaria a tráfico intenso, humo, polvo o combustión."},
        {"value": 7, "label": "Muy alta", "example": "Exposición laboral o residencial intensa a contaminantes."},
        {"value": 8, "label": "Extrema", "example": "Ambiente muy contaminado de forma constante o prolongada."},
    ],
    "Alcohol use": [
        {"value": 1, "label": "Muy bajo o nulo", "example": "No consume alcohol."},
        {"value": 2, "label": "Muy bajo", "example": "Consume en ocasiones muy aisladas, por ejemplo una vez al mes o menos."},
        {"value": 3, "label": "Bajo", "example": "Consume ocasionalmente, alrededor de 1 a 2 veces al mes."},
        {"value": 4, "label": "Regular", "example": "Consume aproximadamente una vez por semana."},
        {"value": 5, "label": "Moderado", "example": "Consume 2 a 3 veces por semana."},
        {"value": 6, "label": "Alto", "example": "Consume 4 a 5 veces por semana."},
        {"value": 7, "label": "Muy alto", "example": "Consume casi todos los días."},
        {"value": 8, "label": "Extremo", "example": "Consume diariamente o en cantidades elevadas."},
    ],
    "Obesity": [
        {"value": 1, "label": "Muy baja o ausente", "example": "Peso normal o sin obesidad aparente. Referencia: IMC menor de 25."},
        {"value": 2, "label": "Baja", "example": "Sobrepeso leve. Referencia aproximada: IMC 25 a 27.4."},
        {"value": 3, "label": "Leve", "example": "Sobrepeso alto. Referencia aproximada: IMC 27.5 a 29.9."},
        {"value": 4, "label": "Moderada", "example": "Obesidad clase I. Referencia aproximada: IMC 30 a 34.9."},
        {"value": 5, "label": "Alta", "example": "Obesidad clase II. Referencia aproximada: IMC 35 a 39.9."},
        {"value": 6, "label": "Muy alta", "example": "Obesidad clase III. Referencia aproximada: IMC igual o mayor de 40."},
        {"value": 7, "label": "Extrema", "example": "Obesidad muy severa o con complicaciones asociadas."},
    ],
    "Genetic Risk": [
        {"value": 1, "label": "Muy bajo o nulo", "example": "Sin antecedentes familiares conocidos de cáncer de pulmón."},
        {"value": 2, "label": "Muy bajo", "example": "Antecedente lejano o poco claro en la familia."},
        {"value": 3, "label": "Bajo", "example": "Un familiar no directo con cáncer de pulmón u otro cáncer relacionado."},
        {"value": 4, "label": "Moderado", "example": "Antecedentes familiares no directos o información familiar sugestiva."},
        {"value": 5, "label": "Alto", "example": "Un familiar directo con cáncer de pulmón. Directo: padre, madre, hermanos o hijos."},
        {"value": 6, "label": "Muy alto", "example": "Varios familiares directos con cáncer de pulmón o diagnóstico a edad temprana."},
        {"value": 7, "label": "Extremo", "example": "Antecedente familiar fuerte o mutación genética conocida asociada a riesgo oncológico."},
    ],
}


def _normalize_payload(payload):
    """Valida y ordena los datos de entrada según las columnas del entrenamiento."""
    if not isinstance(payload, dict):
        raise ValueError("La solicitud debe enviarse en formato JSON válido.")

    missing = [feature for feature in features if feature not in payload]
    if missing:
        readable_missing = [VARIABLE_LABELS.get(feature, feature) for feature in missing]
        raise ValueError(f"Faltan variables requeridas: {', '.join(readable_missing)}")

    clean_data = {}
    for feature in features:
        try:
            value = int(float(payload[feature]))
        except (TypeError, ValueError):
            readable = VARIABLE_LABELS.get(feature, feature)
            raise ValueError(f"La variable '{readable}' debe ser numérica.")

        min_value, max_value = VALID_RANGES[feature]
        if value < min_value or value > max_value:
            readable = VARIABLE_LABELS.get(feature, feature)
            raise ValueError(f"{readable} debe estar entre {min_value} y {max_value}.")
        clean_data[feature] = value

    return pd.DataFrame([clean_data], columns=features)


def _describe_inputs(input_df):
    """Devuelve la interpretación textual de las opciones seleccionadas."""
    descriptions = {}
    row = input_df.iloc[0].to_dict()
    for feature, value in row.items():
        selected = next(
            (item for item in VARIABLE_OPTIONS[feature] if item["value"] == int(value)),
            None,
        )
        descriptions[feature] = {
            "label": VARIABLE_LABELS.get(feature, feature),
            "value": int(value),
            "category": selected["label"] if selected else "No especificado",
            "example": selected["example"] if selected else "",
        }
    return descriptions


def _generate_report(prediction, input_df):
    """Genera un reporte preventivo orientativo según la clase predicha y las entradas."""
    row = input_df.iloc[0].to_dict()

    smoking = int(row.get("Smoking", 1))
    air_pollution = int(row.get("Air Pollution", 1))
    alcohol = int(row.get("Alcohol use", 1))
    obesity = int(row.get("Obesity", 1))
    genetic_risk = int(row.get("Genetic Risk", 1))

    recommendations = []

    if prediction == "Low":
        general_message = (
            "El modelo estima un nivel de riesgo bajo. Se recomienda mantener hábitos saludables, "
            "evitar la exposición al humo del tabaco, reducir la exposición a contaminantes ambientales "
            "y realizar controles preventivos según la orientación de un profesional de salud."
        )
    elif prediction == "Medium":
        general_message = (
            "El modelo estima un nivel de riesgo medio. Se recomienda prestar mayor atención a los factores "
            "modificables, especialmente tabaquismo, consumo de alcohol, exposición a contaminación del aire "
            "y control del peso. Si existen síntomas respiratorios persistentes o antecedentes familiares, "
            "sería conveniente acudir a una evaluación preventiva."
        )
    else:
        general_message = (
            "El modelo estima un nivel de riesgo alto. Este resultado no confirma cáncer de pulmón, "
            "pero sugiere la necesidad de una evaluación médica preventiva, especialmente si existen síntomas "
            "como tos persistente, dificultad para respirar, dolor torácico, pérdida de peso inexplicada "
            "o tos con sangre."
        )

    if smoking >= 6:
        recommendations.append(
            "Tabaquismo elevado: se recomienda buscar apoyo profesional para reducir o abandonar el consumo "
            "de tabaco. Este factor es uno de los más relevantes en la prevención del cáncer de pulmón."
        )
    elif smoking >= 4:
        recommendations.append(
            "Tabaquismo moderado: se recomienda reducir progresivamente la exposición al tabaco y evitar ambientes con humo."
        )

    if air_pollution >= 6:
        recommendations.append(
            "Exposición alta a contaminación del aire: procure reducir la exposición a humo, polvo, tráfico intenso, "
            "combustión o ambientes laborales contaminados. Si corresponde, utilice medidas de protección."
        )
    elif air_pollution >= 4:
        recommendations.append(
            "Exposición moderada a contaminación: se recomienda ventilar los espacios, evitar humo ambiental "
            "y reducir la permanencia en zonas con alta concentración de polvo o tráfico."
        )

    if alcohol >= 6:
        recommendations.append(
            "Consumo de alcohol elevado: se recomienda moderar el consumo y evitar patrones frecuentes o excesivos."
        )
    elif alcohol >= 4:
        recommendations.append(
            "Consumo de alcohol moderado: se sugiere mantener un consumo ocasional y evitar incrementarlo."
        )

    if obesity >= 5:
        recommendations.append(
            "Obesidad elevada: se recomienda fortalecer hábitos de alimentación saludable, actividad física regular "
            "y control del peso con orientación profesional."
        )
    elif obesity >= 4:
        recommendations.append(
            "Obesidad moderada: se recomienda vigilar el peso corporal, mejorar la alimentación y realizar actividad física de forma progresiva."
        )

    if genetic_risk >= 5:
        recommendations.append(
            "Riesgo genético elevado: si existen antecedentes familiares de cáncer de pulmón u otras neoplasias, "
            "se recomienda comentarlo con un profesional de salud para valorar controles preventivos."
        )
    elif genetic_risk >= 3:
        recommendations.append(
            "Riesgo genético moderado: se recomienda tener presente los antecedentes familiares durante los controles médicos."
        )

    if not recommendations:
        recommendations.append(
            "No se identificaron valores elevados en los factores ingresados. Se recomienda mantener controles preventivos y hábitos saludables."
        )

    return {
        "mensaje_general": general_message,
        "recomendaciones_especificas": recommendations,
        "advertencia": (
            "Este reporte es orientativo y se basa en un modelo de machine learning. "
            "No constituye diagnóstico médico, no confirma cáncer de pulmón y no reemplaza la evaluación de un profesional de salud."
        ),
    }


def _predict_from_dataframe(input_df):
    input_scaled = scaler.transform(input_df)
    prediction = str(model.predict(input_scaled)[0])
    probabilities = model.predict_proba(input_scaled)[0]

    probability_result = {
        str(class_label): round(float(probability), 4)
        for class_label, probability in zip(model.classes_, probabilities)
    }

    report = _generate_report(prediction, input_df)

    return {
        "risk_level": prediction,
        "risk_level_es": RISK_TRANSLATION.get(prediction, prediction),
        "probabilities": probability_result,
        "input": input_df.iloc[0].to_dict(),
        "input_descriptions": _describe_inputs(input_df),
        "report": report,
        "note": "Resultado exploratorio de apoyo preventivo; no constituye diagnóstico clínico.",
    }


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        features=features,
        labels=VARIABLE_LABELS,
        ranges=VALID_RANGES,
        options=VARIABLE_OPTIONS,
        variable_images=VARIABLE_IMAGES,
        report_images=REPORT_IMAGES,
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "ExtraTreesClassifier",
        "features": features,
        "classes": list(model.classes_),
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            payload = request.get_json()
        else:
            payload = request.form.to_dict()

        input_df = _normalize_payload(payload)
        result = _predict_from_dataframe(input_df)
        return jsonify(result)

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Error interno: {str(exc)}"}), 500


# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=5000, debug=True)

if __name__ == "__main__":
    app.run(debug=True)