
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.applications import efficientnet_v2

# Caminhos e parâmetros
TRAIN_DIR = 'C:/D9/dataset-desafio/training-specific'
TEST_DIR = 'C:/D9/dataset-desafio/interpretabilidade-specific'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Função para encontrar os pares de imagens e seus rótulos
def create_paired_data_dict(data_dir):
    paired_data = {}
    for label_str in os.listdir(data_dir):
        label = 1 if label_str == 'damaged' else 0
        class_dir = os.path.join(data_dir, label_str)
        for view_type in os.listdir(class_dir):
            view_dir = os.path.join(class_dir, view_type)
            for filename in os.listdir(view_dir):

                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    serial_number = filename.split('_')[0]
                    if serial_number not in paired_data:
                        paired_data[serial_number] = {'label': label}
                    paired_data[serial_number][view_type] = os.path.join(view_dir, filename)
    return paired_data

# Função para carregar e pré-processar um par de imagens
def load_and_preprocess_pair(path_side, path_top, label):
    img_side = tf.io.read_file(path_side)
    img_side = tf.io.decode_png(img_side, channels=3)
    img_side = tf.image.resize(img_side, IMG_SIZE)
    
    img_top = tf.io.read_file(path_top)
    img_top = tf.io.decode_png(img_top, channels=3)
    img_top = tf.image.resize(img_top, IMG_SIZE)
    
    return (img_side, img_top), label

# --- Preparando os dados de Treino e Validação ---
print("Preparando dados de Treino e Validação...")
train_paired_dict = create_paired_data_dict(TRAIN_DIR)

train_paths_side = [data['side'] for sn, data in train_paired_dict.items() if 'side' in data and 'top' in data]
train_paths_top = [data['top'] for sn, data in train_paired_dict.items() if 'side' in data and 'top' in data]
train_labels = [data['label'] for sn, data in train_paired_dict.items() if 'side' in data and 'top' in data]

# Dividindo os dados de treino em treino e validação (80/20)
X_train_side, X_val_side, X_train_top, X_val_top, y_train, y_val = train_test_split(
    train_paths_side, train_paths_top, train_labels, test_size=0.2, random_state=123, stratify=train_labels
)

# Criando os tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_side, X_train_top, y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((X_val_side, X_val_top, y_val))

# --- Preparando os dados de Teste ---
print("Preparando dados de Teste...")
test_paired_dict = create_paired_data_dict(TEST_DIR)
test_paths_side = [data['side'] for sn, data in test_paired_dict.items() if 'side' in data and 'top' in data]
test_paths_top = [data['top'] for sn, data in test_paired_dict.items() if 'side' in data and 'top' in data]
test_labels = [data['label'] for sn, data in test_paired_dict.items() if 'side' in data and 'top' in data]

test_dataset = tf.data.Dataset.from_tensor_slices((test_paths_side, test_paths_top, test_labels))

# --- Criando o pipeline final (map, batch, prefetch) ---
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.map(load_and_preprocess_pair, num_parallel_calls=AUTOTUNE).shuffle(buffer_size=len(y_train)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
validation_dataset = validation_dataset.map(load_and_preprocess_pair, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_dataset = test_dataset.map(load_and_preprocess_pair, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("Pipelines de dados criados com sucesso!")




# ==============================================================================
# 2. NOVA ARQUITETURA DE MODELO (MULTI-INPUT)
# ==============================================================================

# A camada de Augmentation continua a mesma
# Adicionando mais opções de augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(factor=0.2), # Simula diferentes contrastes
    tf.keras.layers.RandomBrightness(factor=0.2), # Simula diferentes iluminações
])

# Vamos criar um "extrator de features" que será compartilhado pelas duas entradas
# Isso é mais eficiente do que ter duas EfficientNets completas
base_model = keras.applications.EfficientNetV2B0(
    weights='imagenet',
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False
)
base_model.trainable = False

# O extrator aplica augmentation, pré-processamento e o modelo base
# O extrator de features com a correção
feature_extractor = tf.keras.Sequential([
    data_augmentation,
    # CORREÇÃO: Usamos a camada Lambda para "transformar" a função em uma camada
    keras.layers.Lambda(efficientnet_v2.preprocess_input),
    base_model,
    keras.layers.GlobalAveragePooling2D()
], name='feature_extractor') # Adicionar um nome é uma boa prática

# Definimos as duas entradas ("olhos" do modelo)
input_side = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name='side_image')
input_top = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name='top_image')

# Processamos cada entrada com o mesmo extrator de features
features_side = feature_extractor(input_side)
features_top = feature_extractor(input_top)

# Concatenamos (juntamos) as features das duas imagens
x = keras.layers.Concatenate()([features_side, features_top])
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dropout(0.3)(x)

# Camada de saída final (problema binário: damage vs intact)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

# Criamos o modelo final, especificando suas duas entradas e uma saída
model = keras.Model(inputs=[input_side, input_top], outputs=outputs)

# Compilamos o modelo (problema binário)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()




# As chamadas de treino e avaliação continuam as mesmas!
history = model.fit(
    train_dataset,
    validation_data = validation_dataset,
    epochs = 20
)

# # ==============================================================================
# # FASE DE FINE-TUNING (VERSÃO SIMPLIFICADA E SEGURA)
# # ==============================================================================
# print("\n--- Iniciando Fase de Fine-Tuning ---")

# # Passo 1: Descongelar o modelo base (ou parte dele)
# base_model.trainable = True
# # Opcional: manter as camadas mais profundas congeladas
# # for layer in base_model.layers[:150]:
# #    layer.trainable = False

# # Passo 2: Recompilar o modelo com uma taxa de aprendizado muito baixa
# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.0001), # Taxa de aprendizado baixa
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# # Passo 3: Continuar o treinamento por um NÚMERO FIXO de novas épocas
# fine_tune_epochs = 10 # Defina quantas épocas de ajuste fino você quer
# print(f"Treinando por mais {fine_tune_epochs} épocas para o ajuste fino...")

# # A chamada .fit agora é mais simples
# history_fine_tune = model.fit(
#     train_dataset,
#     epochs=fine_tune_epochs, # Treina apenas pelo número de épocas de fine-tuning
#     validation_data=validation_dataset
# )

# A avaliação final continua a mesma...
# A avaliação final agora usará o modelo após o fine-tuning
print("\n--- Avaliação Final no Conjunto de Teste (Após Fine-Tuning) ---")
loss, accuracy = model.evaluate(test_dataset)
print(f'Loss final no teste: {loss}')
print(f'Acurácia final no teste: {accuracy}')

# INSIRA ESTE BLOCO APÓS O model.evaluate() FINAL

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

print("\n--- Gerando Relatório Detalhado de Classificação ---")

# Passo 1: Obter os rótulos verdadeiros do test_dataset
y_true = np.concatenate([y for x, y in test_dataset], axis=0)

# Passo 2: Fazer previsões no conjunto de teste completo
# O modelo multi-input espera uma lista ou tupla de inputs
# Como nosso test_dataset já está no formato ((img_side, img_top), label),
# podemos passar ele diretamente para o predict.
y_pred_probs = model.predict(test_dataset)

# Passo 3: Converter as probabilidades em classes preditas
# Como é um problema binário com saída sigmoid, a classe será 1 se a prob > 0.5, senão 0
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Passo 4: Gerar e imprimir o Relatório de Classificação
# Este relatório contém Precision, Recall e F1-Score.
print("\nRelatório de Classificação:\n")
# Os 'target_names' ajudam a identificar o que é a classe 0 e 1
print(classification_report(y_true, y_pred, target_names=['intact (0)', 'damage (1)']))


# Passo 5: Gerar e visualizar a Matriz de Confusão
print("\nGerando Matriz de Confusão...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
            xticklabels=['intact', 'damage'],
            yticklabels=['intact', 'damage'])
plt.xlabel('Previsto', fontsize=13)
plt.ylabel('Verdadeiro', fontsize=13)
plt.title('Matriz de Confusão Final', fontsize=15)
plt.show()

import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import matplotlib.pyplot as plt

def read_bbox_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))
        return [xmin, ymin, xmax, ymax]
    return None

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")

def generate_gradcam(model, img, layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img])
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
    
    heatmap = tf.maximum(cam, 0) / tf.reduce_max(cam)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))
    return heatmap

def heatmap_to_bbox(heatmap, threshold=0.4):
    binary_mask = heatmap > threshold
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return [x, y, x + w, y + h]
    return [0, 0, 0, 0]

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate_model_with_cam(model, side_dir, threshold=0.4):
    layer_name = get_last_conv_layer(model)
    ious = []

    for fname in os.listdir(side_dir):
        if not fname.endswith('.png'):
            continue

        img_path = os.path.join(side_dir, fname)
        xml_path = img_path.replace(".png", ".xml")
        if not os.path.exists(xml_path):
            continue

        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_input = tf.expand_dims(img_array, axis=0)

        heatmap = generate_gradcam(model, img_input, layer_name)
        pred_box = heatmap_to_bbox(heatmap, threshold)
        gt_box = read_bbox_from_xml(xml_path)
        iou = compute_iou(pred_box, gt_box)
        ious.append(iou)

        # Visualização (opcional)
        img_np = img_array.astype(np.uint8)
        overlay = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)
        cv2.rectangle(overlay, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0), 2)
        cv2.rectangle(overlay, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 0, 255), 2)

        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.title(f"IoU: {iou:.2f}")
        plt.axis("off")
        plt.show()

    print(f"[INFO] Média final de IoU: {np.mean(ious):.4f}")
    print(f"IoU acima de 0.5: {np.sum(np.array(ious) > 0.5)} / {len(ious)}")

# Exemplo de uso
side_images_path = os.path.join(TEST_DIR, "damage/side")  # ajuste se necessário
evaluate_model_with_cam(model, side_images_path)
