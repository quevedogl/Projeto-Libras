import os
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = "data"
IMG_SIZE = 224
BATCH = 16
EPOCHS = 8
VAL_FRACTION = 0.20
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class_names = sorted(
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
    and any(f.lower().endswith(".jpg") for f in os.listdir(os.path.join(DATA_DIR, d)))
)
if len(class_names) < 2:
    raise SystemExit("Poucas classes com fotos. Execute collect.py e colete as imagens primeiro.")

train_paths, train_y, val_paths, val_y = [], [], [], []
for idx, cls in enumerate(class_names):
    files = sorted(
        os.path.join(DATA_DIR, cls, f)
        for f in os.listdir(os.path.join(DATA_DIR, cls))
        if f.lower().endswith(".jpg")
    )
    if len(files) < 5:
        raise SystemExit(f"A classe '{cls}' tem apenas {len(files)} fotos. Colete pelo menos 5; idealmente muito mais.")
    cut = max(1, int(len(files) * (1 - VAL_FRACTION)))
    train_paths += files[:cut]
    train_y += [idx] * cut
    val_paths += files[cut:]
    val_y += [idx] * (len(files) - cut)
    print(f"{cls}: {cut} treino / {len(files) - cut} validação")

def load_image(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, label

def make_dataset(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(max(1, len(paths)), seed=SEED)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_paths, train_y, True)
val_ds = make_dataset(val_paths, val_y, False)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomBrightness(0.20),
    tf.keras.layers.RandomContrast(0.20),
])

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
model.save("gesture_model.keras")
with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

plt.figure(figsize=(10, 4))
for i, (metric, title) in enumerate([("accuracy", "Acurácia"), ("loss", "Loss")]):
    plt.subplot(1, 2, i + 1)
    plt.plot(history.history[metric], label="treino")
    plt.plot(history.history["val_" + metric], label="validação")
    plt.title(title)
    plt.xlabel("Época")
    plt.legend()
plt.tight_layout()
plt.savefig("curvas.png")
plt.close()

y_true = np.array(val_y)
y_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
cm = np.zeros((len(class_names), len(class_names)), dtype=int)
for true, pred in zip(y_true, y_pred):
    cm[true, pred] += 1
print("\nMatriz de confusão (linhas = real; colunas = previsto):")
print("          " + " ".join(f"{c[:8]:>8}" for c in class_names))
for i, cls in enumerate(class_names):
    print(f"{cls[:10]:>10} " + " ".join(f"{v:>8}" for v in cm[i]))
print("\nModelo salvo: gesture_model.keras")

def generate_dataset_pdf(data_dir="data", output="dataset_imagens.pdf", cols=8, quality=70):
    page_w, page_h, margin, top = 1240, 1754, 40, 110
    cell = (page_w - 2 * margin) // cols
    rows = (page_h - top - margin) // cell
    per_page = cols * rows
    try:
        font = ImageFont.load_default(size=30)
    except TypeError:  # Pillow mais antigo
        font = ImageFont.load_default()
    pages = []
    classes = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    for cls in classes:
        folder = os.path.join(data_dir, cls)
        photos = sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg"))
        if not photos:
            continue
        n_claro = sum(1 for f in photos if f.lower().endswith("_claro.jpg"))
        n_escuro = sum(1 for f in photos if f.lower().endswith("_escuro.jpg"))
        luz = f" (fundo claro: {n_claro} | fundo escuro: {n_escuro})" if (n_claro or n_escuro) else ""
        total_pages = max(1, (len(photos) + per_page - 1) // per_page)
        for page_idx in range(total_pages):
            page = Image.new("RGB", (page_w, page_h), "white")
            draw = ImageDraw.Draw(page)
            draw.text((margin, 40), f"Classe: {cls} | {len(photos)} imagens{luz} | pag {page_idx + 1}/{total_pages}",
                      fill="black", font=font)
            batch = photos[page_idx * per_page:(page_idx + 1) * per_page]
            for i, filename in enumerate(batch):
                path = os.path.join(folder, filename)
                try:
                    with Image.open(path) as source:
                        img = source.convert("RGB")
                        img.thumbnail((cell - 6, cell - 6))
                        x = margin + (i % cols) * cell + 3
                        y = top + (i // cols) * cell + 3
                        page.paste(img, (x, y))
                except Exception as exc:
                    print(f"Não consegui abrir {path}: {exc}")
            pages.append(page)
    if not pages:
        raise SystemExit("Não há imagens para colocar no PDF.")
    pages[0].save(output, save_all=True, append_images=pages[1:],
                  quality=quality, resolution=150)
    print(f"PDF criado: {output} ({len(pages)} páginas)")

generate_dataset_pdf()
