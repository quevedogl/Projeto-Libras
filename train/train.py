import os
import json
import random

import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURAÇÕES
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
VAL_FRACTION = 0.2
SEED = 42

MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "class_names.json")
CHART_PATH = os.path.join(BASE_DIR, "curvas.png")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# VERIFICAÇÃO DOS DADOS
# =========================================================

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"A pasta de dados não foi encontrada: {DATA_DIR}"
    )

class_names = sorted(
    folder
    for folder in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, folder))
)

if len(class_names) < 2:
    raise ValueError(
        "São necessárias pelo menos duas classes para treinar o modelo."
    )

print("\n========================================")
print(" TREINAMENTO DO MODELO DE GESTOS")
print("========================================")
print("Classes encontradas:", class_names)


# =========================================================
# SEPARAÇÃO DOS DADOS
# =========================================================

train_paths = []
train_labels = []

val_paths = []
val_labels = []

for class_index, class_name in enumerate(class_names):

    class_dir = os.path.join(DATA_DIR, class_name)

    files = sorted(
        os.path.join(class_dir, filename)
        for filename in os.listdir(class_dir)
        if filename.lower().endswith(".jpg")
    )

    if len(files) < 2:
        raise ValueError(
            f"A classe '{class_name}' precisa de pelo menos "
            "duas imagens para separar treino e validação."
        )

    # Mantém a separação temporal usada no código original.
    split_index = int(len(files) * (1 - VAL_FRACTION))

    # Garante pelo menos uma imagem em cada conjunto.
    split_index = max(1, min(split_index, len(files) - 1))

    train_files = files[:split_index]
    validation_files = files[split_index:]

    train_paths.extend(train_files)
    train_labels.extend([class_index] * len(train_files))

    val_paths.extend(validation_files)
    val_labels.extend([class_index] * len(validation_files))

    print(
        f"{class_name}: "
        f"{len(train_files)} imagens para treino / "
        f"{len(validation_files)} para validação"
    )

if not train_paths or not val_paths:
    raise ValueError(
        "Os conjuntos de treino e validação não podem estar vazios."
    )


# =========================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# =========================================================

def load_image(image_path, label):
    """Carrega a imagem, decodifica o JPG e redimensiona."""

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    return image, label


def create_dataset(paths, labels, training=False):
    """Cria um pipeline eficiente de carregamento de imagens."""

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_ds = create_dataset(
    train_paths,
    train_labels,
    training=True
)

val_ds = create_dataset(
    val_paths,
    val_labels,
    training=False
)


# =========================================================
# AUMENTO DE DADOS
# =========================================================

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomBrightness(0.25),
        tf.keras.layers.RandomContrast(0.25),
    ],
    name="data_augmentation"
)


# =========================================================
# MODELO MOBILENETV2
# =========================================================

print("\nCarregando MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Congela as camadas pré-treinadas.
base_model.trainable = False


# =========================================================
# CONSTRUÇÃO DO MODELO
# =========================================================

inputs = tf.keras.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3),
    name="input_image"
)

x = augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax",
    name="gesture_prediction"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="gesture_classifier"
)


# =========================================================
# COMPILAÇÃO
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nResumo do modelo:")
model.summary()


# =========================================================
# TREINAMENTO
# =========================================================

print("\nIniciando treinamento...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


# =========================================================
# SALVAMENTO DO MODELO
# =========================================================

model.save(MODEL_PATH)

with open(
    CLASSES_PATH,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        class_names,
        file,
        ensure_ascii=False,
        indent=4
    )

print(f"\nModelo salvo em: {MODEL_PATH}")
print(f"Classes salvas em: {CLASSES_PATH}")


# =========================================================
# GRÁFICOS DE TREINAMENTO
# =========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)

axes[0].plot(
    history.history["accuracy"],
    label="Treino"
)

axes[0].plot(
    history.history["val_accuracy"],
    label="Validação"
)

axes[0].set_title("Acurácia")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Acurácia")
axes[0].legend()


axes[1].plot(
    history.history["loss"],
    label="Treino"
)

axes[1].plot(
    history.history["val_loss"],
    label="Validação"
)

axes[1].set_title("Perda (Loss)")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Loss")
axes[1].legend()


fig.tight_layout()
fig.savefig(CHART_PATH, dpi=150)
plt.close(fig)

print(f"Gráficos salvos em: {CHART_PATH}")


# =========================================================
# MATRIZ DE CONFUSÃO
# =========================================================

print("\nCalculando matriz de confusão...")

predictions = model.predict(
    val_ds,
    verbose=0
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

true_labels = np.asarray(
    val_labels,
    dtype=int
)

confusion_matrix = tf.math.confusion_matrix(
    true_labels,
    predicted_labels,
    num_classes=len(class_names)
).numpy()

print(
    "\nMatriz de confusão "
    "(linhas = classe real; colunas = classe prevista):"
)

print(
    " " * 12
    + " ".join(f"{name[:10]:>10}" for name in class_names)
)

for index, class_name in enumerate(class_names):

    values = " ".join(
        f"{value:>10}"
        for value in confusion_matrix[index]
    )

    print(f"{class_name[:10]:>12} {values}")


# =========================================================
# RESULTADO FINAL
# =========================================================

print("\n========================================")
print(" TREINAMENTO CONCLUÍDO")
print("========================================")

print(f"Total de classes: {len(class_names)}")
print(f"Imagens de treino: {len(train_paths)}")
print(f"Imagens de validação: {len(val_paths)}")

print(f"\nModelo: {MODEL_PATH}")
print(f"Classes: {CLASSES_PATH}")
print(f"Gráficos: {CHART_PATH}")

print("\nPróximo passo: executar o programa de reconhecimento.")
