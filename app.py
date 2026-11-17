"""
Web Application & REST API Service
Project: AI-Based Eye Disease Prediction and Clinical Decision Support System
"""

import os
import json
from flask import Flask, request, jsonify, render_template
from src.inference import EyeDiseasePredictor

app = Flask(__name__, template_folder='templates')

# Initialize the inference engine on startup
MODEL_PATH = os.path.join('saved_models', 'eye_disease_pipeline.joblib')
METADATA_PATH = os.path.join('saved_models', 'model_metadata.json')

try:
    predictor = EyeDiseasePredictor(model_path=MODEL_PATH, metadata_path=METADATA_PATH)
    print("Inference Engine successfully initialized.")
except Exception as e:
    print(f"Warning: Could not initialize model ({e}). Ensure src/train.py has been executed.")
    predictor = None


@app.route('/')
def home():
    """Serves the interactive clinical decision support web dashboard."""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Service health and model readiness check."""
    status = "healthy" if predictor is not None else "model_not_loaded"
    return jsonify({
        "status": status,
        "model_loaded": predictor is not None,
        "version": "1.0.0",
        "service": "Myopsia Clinical Decision Support System"
    })


@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Returns model training metadata, target classes, and feature lists."""
    if predictor and predictor.metadata:
        return jsonify(predictor.metadata)
    return jsonify({"error": "Metadata not available"}), 404


@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    """
    REST API endpoint for real-time patient assessment.
    Accepts JSON containing patient clinical parameters and symptoms.
    """
    if predictor is None:
        return jsonify({"error": "Inference engine is not initialized"}), 503
        
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400
            
        result = predictor.predict(data)
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    # Default local development server
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Myopsia Web Application at http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False)
