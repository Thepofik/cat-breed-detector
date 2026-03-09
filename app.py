# ============================================
# ОПРЕДЕЛИТЕЛЬ ПОРОД КОШЕК - ПОЛНАЯ ВЕРСИЯ
# ============================================

# ---------- БИБЛИОТЕКИ ----------
from flask import Flask, request, render_template
import os
import numpy as np
import json
import time
import random
from werkzeug.utils import secure_filename

# ---------- СОЗДАНИЕ ПРИЛОЖЕНИЯ ----------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------- ЗАГРУЗКА ПОРОД ----------
print('🔄 Загрузка списка пород...')
try:
    with open('model/class_names.json', 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    print(f'✅ Загружено пород: {len(class_names)}')
except Exception as e:
    print(f'❌ Ошибка: {e}')
    class_names = {str(i): f'Порода {i + 1}' for i in range(10)}

# ---------- ЗАГРУЗКА ОПИСАНИЙ ПОРОД ----------
print('🔄 Загрузка описаний пород...')
try:
    with open('model/breed_descriptions.json', 'r', encoding='utf-8') as f:
        breed_descriptions = json.load(f)
    print(f'✅ Загружено описаний: {len(breed_descriptions)}')
except Exception as e:
    print(f'⚠️ Описания не загружены: {e}')
    breed_descriptions = {}

# ---------- ПРОВЕРКА TENSORFLOW ----------
print('🔄 Проверка TensorFlow...')
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    TENSORFLOW_AVAILABLE = True
    print('✅ TensorFlow доступен')
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    print('⚠️ TensorFlow НЕ доступен (демо-режим)')
    print(f'   Ошибка: {e}')

# ---------- ЗАГРУЗКА МОДЕЛИ ----------
model = None
if TENSORFLOW_AVAILABLE:
    model_path = 'model/cat_breed_model.h5'
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print('✅ Модель нейросети загружена')
            print(f'   Размер: {os.path.getsize(model_path) / 1024 / 1024:.1f} МБ')
        except Exception as e:
            print(f'❌ Ошибка загрузки модели: {e}')
            model = None
    else:
        print('⚠️ Файл модели не найден. Будет использован демо-режим')
else:
    print('⚠️ TensorFlow недоступен, пропускаем загрузку модели')


# ---------- ФУНКЦИИ ----------
def allowed_file(filename):
    """Проверка расширения файла"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in app.config['ALLOWED_EXTENSIONS']


def prepare_image(img_path):
    """Подготовка изображения для нейросети"""
    if not TENSORFLOW_AVAILABLE:
        return np.random.random((1, 224, 224, 3))

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_breed(img_path):
    """Определение породы - возвращает ТОЛЬКО одну лучшую породу с описанием"""

    # Если нет модели или TensorFlow - демо-режим
    if model is None or not TENSORFLOW_AVAILABLE:
        print('   Демо-режим: случайное предсказание')
        all_breeds = list(class_names.values())
        selected = random.choice(all_breeds)
        confidence = random.uniform(70, 98)

        # Получаем описание для выбранной породы
        description = breed_descriptions.get(selected,
                                             "Описание для этой породы временно отсутствует. Мы работаем над его добавлением.")

        return {
            'success': True,
            'demo': True,
            'predictions': [{
                'breed': selected,
                'confidence': round(confidence, 1),
                'description': description
            }]
        }

    # Реальный режим с нейросетью
    try:
        print('   Реальный режим: обработка нейросетью')
        img_array = prepare_image(img_path)
        predictions = model.predict(img_array, verbose=0)[0]

        # Находим индекс самой вероятной породы
        best_idx = np.argmax(predictions)
        best_breed = class_names.get(str(best_idx), f'Неизвестная порода')
        best_confidence = float(predictions[best_idx] * 100)

        # Получаем описание для найденной породы
        description = breed_descriptions.get(best_breed,
                                             "Описание для этой породы временно отсутствует. Мы работаем над его добавлением.")

        print(f'   Лучшая порода: {best_breed} с уверенностью {best_confidence:.1f}%')

        return {
            'success': True,
            'demo': False,
            'predictions': [{
                'breed': best_breed,
                'confidence': round(best_confidence, 1),
                'description': description
            }]
        }

    except Exception as e:
        print(f'❌ Ошибка: {e}')
        return {'success': False, 'error': str(e)}


# ---------- МАРШРУТЫ ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    """Главная страница"""
    if request.method == 'POST':
        if 'photo' not in request.files:
            return render_template('index.html', error='Нет файла')

        file = request.files['photo']

        if file.filename == '':
            return render_template('index.html', error='Файл не выбран')

        if not allowed_file(file.filename):
            return render_template('index.html', error='Можно только JPG, PNG, GIF')

        # Сохраняем файл
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename = f'{timestamp}_{filename}'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Определяем породу
        result = predict_breed(filepath)

        # Удаляем файл
        try:
            os.remove(filepath)
        except:
            pass

        if result['success']:
            return render_template('result.html',
                                   results=result['predictions'],
                                   demo=result.get('demo', False))
        else:
            return render_template('index.html', error='Ошибка обработки')

    return render_template('index.html')


@app.route('/health')
def health():
    """Проверка работы"""
    return {
        'status': 'ok',
        'tensorflow': TENSORFLOW_AVAILABLE,
        'model': model is not None,
        'breeds': len(class_names),
        'descriptions': len(breed_descriptions)
    }


# ---------- ЗАПУСК ----------
if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('🐱 ОПРЕДЕЛИТЕЛЬ ПОРОД КОШЕК ЗАПУЩЕН')
    print('=' * 60)
    print(f'📁 Папка: {app.config["UPLOAD_FOLDER"]}')
    print(f'📊 TensorFlow: {"✅ ДОСТУПЕН" if TENSORFLOW_AVAILABLE else "⚠️ НЕ ДОСТУПЕН"}')
    print(f'🤖 Режим: {"✅ РЕАЛЬНАЯ НЕЙРОСЕТЬ" if model is not None else "⚠️ ДЕМО-РЕЖИМ"}')
    print(f'📚 Пород: {len(class_names)}')
    print(f'📖 Описаний: {len(breed_descriptions)}')
    print('=' * 60)
    print('🌐 Откройте в браузере:')
    print('👉 http://127.0.0.1:5000')
    print('=' * 60 + '\n')

    app.run(debug=True, host='127.0.0.1', port=5000)