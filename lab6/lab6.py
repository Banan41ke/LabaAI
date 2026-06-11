import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Dense, Flatten, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,  # Повороты изображений
    width_shift_range=0.1,  # Сдвиги по ширине
    height_shift_range=0.1,  # Сдвиги по высоте
    horizontal_flip=True,  # Горизонтальные отражения
    validation_split=0.2,  # 20% данных на валидацию
)

train_data = train_datagen.flow_from_directory(
    "dataset/",
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="training",
)

val_data = train_datagen.flow_from_directory(
    "dataset/",
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
)

num_classes = len(train_data.class_indices)
class_labels = {v: k for k, v in train_data.class_indices.items()}

custom_model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),  # Дополнительный слой
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ]
)

custom_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
print("=== Обучение кастомной CNN ===")
custom_model.fit(train_data, validation_data=val_data, epochs=5)

base_mobilenet = MobileNetV2(
    input_shape=(64, 64, 3), include_top=False, weights="imagenet"
)
base_mobilenet.trainable = False  # Замораживаем веса базовой сети

mobilenet_model = Sequential(
    [
        base_mobilenet,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ]
)

mobilenet_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
print("\n=== Обучение модели на базе MobileNetV2 ===")
mobilenet_model.fit(train_data, validation_data=val_data, epochs=5)

print("\n=== Сравнение результатов ===")
loss_c, acc_c = custom_model.evaluate(val_data, verbose=0)
loss_m, acc_m = mobilenet_model.evaluate(val_data, verbose=0)
print(f"Точность кастомной CNN: {acc_c * 100:.2f}%")
print(f"Точность MobileNetV2: {acc_m * 100:.2f}%")

test_img_path = "test_gesture.jpg"

if os.path.exists(test_img_path):
    img = image.load_img(test_img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Используем лучшую модель (например, кастомную) для предсказания
    prediction = custom_model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(prediction)
    predicted_label = class_labels[predicted_class_idx]
    confidence = np.max(prediction)

    print(f"\n=== Анализ тестового снимка ({test_img_path}) ===")
    print(f"Распознанный жест: {predicted_label}")
    print(f"Уверенность сети: {confidence:.2%}")
else:
    print(
        f"\n[Предупреждение]: Файл '{est_img_path}' не найден. "
        f"Сделайте фото руки, назовите его '{test_img_path}' и положите рядом со скриптом."
    )
