"""
STEP 2 — TRAIN EMOTION DETECTION MODEL
MobileNetV2 transfer learning on FER2013.
~30-60 mins on CPU, ~10 mins on GPU.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

DATA_DIR = "../data/fer2013"
MODEL_DIR   = "../models"
IMG_SIZE    = 96
BATCH_SIZE  = 32
EPOCHS      = 50
NUM_CLASSES = 7

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

os.makedirs(MODEL_DIR, exist_ok=True)


def get_generators():
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )
    val_data = val_gen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    return train_data, val_data


def build_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"✅ Model built. Total params: {model.count_params():,}")
    return model, base_model


def train(model, train_data, val_data):
    print("\n🔵 Phase 1 — Training classifier head...")
    callbacks = [
        ModelCheckpoint(f"{MODEL_DIR}/emotion_model_best.h5",
                        monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=7,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    h1 = model.fit(train_data, validation_data=val_data, epochs=25, callbacks=callbacks)

    print("\n🟠 Phase 2 — Fine-tuning top layers...")
    for layer in model.layers[-30:]:
        layer.trainable = True
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks[0] = ModelCheckpoint(f"{MODEL_DIR}/emotion_model_finetuned.h5",
                                   monitor='val_accuracy', save_best_only=True, verbose=1)
    h2 = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)

    return h1, h2


def plot_history(h1, h2):
    acc  = h1.history['accuracy']  + h2.history['accuracy']
    val  = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    vloss= h1.history['val_loss'] + h2.history['val_loss']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(acc, label='Train'); axes[0].plot(val, label='Val')
    axes[0].set_title('Accuracy'); axes[0].legend()
    axes[1].plot(loss, label='Train'); axes[1].plot(vloss, label='Val')
    axes[1].set_title('Loss'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/training_history.png", dpi=150)
    print(f"📊 Plot saved to {MODEL_DIR}/training_history.png")


if __name__ == "__main__":
    print("=" * 50)
    print("  STEP 2 — EMOTION MODEL TRAINING")
    print("=" * 50)

    if not os.path.exists(os.path.join(DATA_DIR, "train")):
        print("❌ Run step1_prepare_data.py first.")
        exit()

    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    train_data, val_data = get_generators()
    model, base_model    = build_model()
    h1, h2               = train(model, train_data, val_data)
    plot_history(h1, h2)

    model.save(f"{MODEL_DIR}/emotion_model_final.h5")
    print(f"\n✅ Model saved. Run step3_inference.py next.")