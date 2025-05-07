import tensorflow as tf
import os

# === Verificar uso de GPU ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU detectada:", gpus[0])
else:
    print("⚠️ GPU no detectada, usando CPU")

# === Generadores de datos ===
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'archive/train',
    target_size=(48, 48),
    batch_size=128,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'archive/test',
    target_size=(48, 48),
    batch_size=128,
    class_mode='categorical'
)

# === Modelo base optimizado ===
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(48, 48, 3),
    include_top=False,
    weights='imagenet',
    alpha=0.5  # Modelo más ligero
)
base_model.trainable = False  # Fase 1: congelado

# === Modelo completo ===
inputs = tf.keras.Input(shape=(48, 48, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(7, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# === Compilar modelo ===
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# === Callback: EarlyStopping ===
early_stop = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)

# === Fase 1: entrenar solo la cabeza ===
model.fit(train_generator, epochs=15, validation_data=test_generator, callbacks=[early_stop])

# === Fase 2: fine-tuning (últimas 30 capas entrenables) ===
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompilar con tasa de aprendizaje reducida
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar fase 2
model.fit(train_generator, epochs=20, validation_data=test_generator, callbacks=[early_stop])

# === Guardar modelo final ===
model.save("modelo_emociones_mobilenetv2_optimo.h5")

