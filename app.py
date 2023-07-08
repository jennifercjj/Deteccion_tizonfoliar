from PIL import Image, ImageDraw, ImageFont
import io
import os
import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)
app.static_folder = 'static'
RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.debug = False

# Constantes
BOUNDING_BOX_COLOR = (255, 0, 0)
LABEL_BACKGROUND_COLOR = (255, 0, 0)
LABEL_TEXT_COLOR = (255, 255, 255)
MIN_FONT_SIZE = 15
MAX_FONT_SIZE = 17

@app.errorhandler(404)
def page_not_found(error):
    return "Página no encontrada", 404

def find_model():
    for f in os.listdir():
        if f.endswith(".pt"):
            return f
    print("Por favor, coloca un archivo de modelo en este directorio!")

# Descargar y cargar el modelo
model_name = find_model()
model = torch.hub.load("WongKinYiu/yolov7", 'custom', model_name)
model.conf = 0.4  # Umbral de confianza

def get_prediction(img_bytes, model):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((640, 640))
    imgs = [img]  # lista de imágenes en batch
    # Realiza la inferencia
    results = model(imgs, size=640)  # incluye NMS
    
    # Diccionario de etiquetas de clase
    class_labels = results.names
    
    # Cambia el color del bounding box a rojo y muestra el nombre de la clase y el porcentaje de detección
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    score2 = 0.0
    for i, result in enumerate(results.pred):
        for box, class_id, score in zip(result[:, :4], result[:, 5].tolist(), result[:, 4].tolist()):
            x1, y1, x2, y2 = box
            class_name = class_labels[int(class_id)]
            score2 = score
            label = f"Tizón Foliar: {score*100:.1f}%"
            
            # Ajustar el tamaño del label según el texto
            font_size = MAX_FONT_SIZE
            font = ImageFont.truetype("arial.ttf", font_size)
            label_width, label_height = draw.textsize(label, font=font)
            
            while label_width > (x2 - x1):
                font_size -= 1
                if font_size < MIN_FONT_SIZE:
                    break
                font = ImageFont.truetype("arial.ttf", font_size)
                label_width, label_height = draw.textsize(label, font=font)
            
            draw.rectangle([(x1, y1), (x2, y2)], outline=BOUNDING_BOX_COLOR, width=3)
            draw.rectangle([(x1, y1), (x1 + label_width, y1 - label_height - 4)], fill=LABEL_BACKGROUND_COLOR)
            draw.text((x1, y1 - label_height - 4), label, fill=LABEL_TEXT_COLOR, font=font)

    return img, score2

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
            
        img_bytes = file.read()
        result_image, score = get_prediction(img_bytes, model)
        
        filename = 'image0.jpg'
        result_image.save(os.path.join(app.config['RESULT_FOLDER'], filename))
        
        return render_template('result.html', result_image=filename, model_name=model_name, score=score)

    return render_template('index.html')

if __name__ == '__main__':
    app.run()
