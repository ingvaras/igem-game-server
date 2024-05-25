from io import BytesIO
import base64
import ssl
import os
from flask import Flask, request
from PIL import Image
import clip
import torch

def decode_and_write_file(env_variable, file_path):
    encoded_data = os.getenv(env_variable)
    if not encoded_data:
        raise ValueError(f"Environment variable {env_variable} is not set")
    decoded_data = base64.b64decode(encoded_data)
    with open(file_path, 'wb') as file:
        file.write(decoded_data)

server_key_path = 'igem-game-server.key'
server_cert_path = 'igem-game-server.crt'
ca_cert_path = 'igem-game-ca.crt'

decode_and_write_file('SERVER_KEY', server_key_path)
decode_and_write_file('SERVER_CERT', server_cert_path)
decode_and_write_file('CA_CERT', ca_cert_path)

app = Flask(__name__)

classes = ["There is skin in the picture", "There is water in the picture", "There is wood in the picture"]
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)
text = clip.tokenize(classes).to(device)

@app.route('/', methods=['POST'])
def predict_surface():
    image_data = request.data
    image_pil = Image.open(BytesIO(image_data))
    image_tensor = preprocess(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image_tensor, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print(classes[probs[0].argmax()])
        return classes[probs[0].argmax()]

def create_ssl_context():
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile=server_cert_path, keyfile=server_key_path)
    ssl_context.load_verify_locations(cafile=ca_cert_path)
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    return ssl_context

if __name__ == '__main__':
    ssl_context = create_ssl_context()
    app.run(host='0.0.0.0', port=5000, ssl_context=ssl_context)
