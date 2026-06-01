from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw
import io
import random

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@app.route("/ping", methods=["GET"])
def ping():
    return "ok", 200

@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(force=True)
    prompt = data.get("prompt", "")

    img = Image.new(
        "RGB",
        (800, 500),
        color=(
            random.randint(40, 180),
            random.randint(40, 180),
            random.randint(40, 180),
        )
    )

    draw = ImageDraw.Draw(img)
    draw.text((40, 220), prompt[:80], fill="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
