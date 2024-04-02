from io import BytesIO

from flask import Flask, request
from PIL import Image
import clip
import torch

app = Flask(__name__)

classes = ["There is skin in the picture", "There is water in the picture", "There is wood in the picture"]
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.route('/', methods=['POST'])
def predict_surface():
    model, preprocess = clip.load("RN50", device=device)
    text = clip.tokenize(classes).to(device)
    image_data = request.data
    image_pil = Image.open(BytesIO(image_data))
    image_tensor = preprocess(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image_tensor, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print(classes[probs[0].argmax()])
        return classes[probs[0].argmax()]
