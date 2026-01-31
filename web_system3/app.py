from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from inference_core import InferenceCore

app = Flask(__name__)
CORS(app)

core = None
try:
    core = InferenceCore()
except Exception as e:
    print(f"Error initializing model: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global core
    if not core:
        return jsonify({'status': 'error', 'message': 'Model not initialized'}), 500

    try:
        data = request.json
        signal_in = np.array(data['signal'])
        fs_in = data.get('fs', 250)

        if len(signal_in) == 0:
            return jsonify({'status': 'error', 'message': 'Empty signal'}), 400

        result = core.predict_from_signal(signal_in, fs=fs_in)

        if isinstance(result, tuple) and len(result) == 2:
            fecg_output, peaks = result
        else:
            fecg_output = result
            peaks = np.array([])

        return jsonify({
            'status': 'success',
            'fecg': fecg_output.tolist(),
            'peaks': peaks.tolist(),
            'fecg_fs': 200  # [新增] 明确告知前端 FECG 是 200Hz
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    print("🚀 DIFF-FECG System Starting at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)