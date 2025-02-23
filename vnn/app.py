from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision import models, transforms
import os

app = Flask(__name__)

# Загрузка предобученной модели ResNet50
model = models.resnet50(pretrained=True)
model.eval()

# Загрузка названий классов ImageNet
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# Преобразование изображения для модели
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Функция классификации изображения
def predict_image(image):
    try:
        img = Image.open(image).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, top_class = probabilities.topk(1, dim=0)
            return classes[top_class.item()], top_prob.item()
    except Exception:
        return None, None


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "Файл не загружен"
        else:
            file = request.files["file"]
            if file.filename == "":
                error = "Файл не выбран"
            else:
                # Проверка формата файла
                allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
                if not os.path.splitext(file.filename)[1].lower() in allowed_extensions:
                    error = "Формат файла не поддерживается. Используйте PNG, JPEG или BMP"
                else:
                    label, prob = predict_image(file)
                    if label:
                        result = label
                        probability = round(prob * 100, 2)
                    else:
                        error = "Не удалось распознать объект на изображении"

    return render_template("index.html", result=result, probability=probability, error=error)


if __name__ == "__main__":
    app.run(debug=True)
