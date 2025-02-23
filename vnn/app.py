from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision import models, transforms
import os

app = Flask(__name__)

# Функция для загрузки модели по имени
def load_model(model_name):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
    else:
        raise ValueError("Неизвестная модель")
    model.eval()
    return model

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
def predict_image(image, model):
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

# основная страница с функционалом
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    error = None
    # Получаем модель из формы или используем "resnet50" по умолчанию
    selected_model = request.form.get("model", "resnet50")

    if request.method == "POST":
        if "file" not in request.files:
            error = "Файл не загружен"
        else:
            file = request.files["file"]
            if file.filename == "":
                error = "Файл не выбран"
            else:
                allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
                if not os.path.splitext(file.filename)[1].lower() in allowed_extensions:
                    error = "Формат файла не поддерживается. Используйте PNG, JPEG или BMP"
                else:
                    try:
                        model = load_model(selected_model)
                        label, prob = predict_image(file, model)
                        if label:
                            result = label
                            probability = round(prob * 100, 2)
                        else:
                            error = "Не удалось распознать объект на изображении"
                    except ValueError as e:
                        error = str(e)

    # Передаем selected_model в шаблон всегда
    return render_template("index.html", result=result, probability=probability, error=error, selected_model=selected_model)

if __name__ == "__main__":
    app.run(debug=True)