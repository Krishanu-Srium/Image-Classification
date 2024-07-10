# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from util import classify_image, load_saved_artifacts

app = Flask(__name__)
CORS(app)

@app.route('/classify_image', methods=['POST'])
def classify_image_endpoint():
    try:
        print("Request received")
        image_data = request.form['image_data']
        print("Image data received")
        result = classify_image(image_base64_data=image_data)
        print("Image classified")
        return jsonify(result)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Loading saved artifacts...start")
    load_saved_artifacts()
    print("Loading saved artifacts...done")
    app.run(port=5500,debug=True)
    
