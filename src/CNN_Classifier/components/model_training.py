import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
import numpy as np

from CNN_Classifier.entity.config_entity import TrainingConfig
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)


# ============================
# TRAINING CLASS
# ============================
class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    # ----------------------------
    # Load Model
    # ----------------------------
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    # ----------------------------
    # Data Generators
    # ----------------------------
    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            preprocessing_function=preprocess_input,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagen = ImageDataGenerator(**datagenerator_kwargs)
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.15,
            shear_range=0.1,
            brightness_range=[0.8, 1.2],
            horizontal_flip=True,
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagen.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        self.train_generator = train_datagen.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    # ----------------------------
    # Compile Model Helper
    # ----------------------------
    def _compile_model(self, lr):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc")
            ]
        )

    # ----------------------------
    # Training
    # ----------------------------
    def train(self):
        # ---- Class Weights ----
        classes = self.train_generator.classes
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(classes),
            y=classes
        )
        class_weight_dict = dict(enumerate(class_weights))

        callbacks = [
            ModelCheckpoint(
                filepath=self.config.trained_model_path,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
            TensorBoard(log_dir="logs/fit")
        ]

        # ============================
        # PHASE 1: Train Classifier Head
        # ============================
        print("\n🔵 Phase 1: Training classifier head")

        for layer in self.model.layers:
            layer.trainable = False

        self._compile_model(lr=1e-4)

        self.model.fit(
            self.train_generator,
            epochs=5,
            validation_data=self.valid_generator,
            callbacks=callbacks,
            class_weight=class_weight_dict
        )

        # ============================
        # PHASE 2: Fine-tuning Backbone
        # ============================
        print("\n🟢 Phase 2: Fine-tuning backbone")

        for layer in self.model.layers[-50:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

        self._compile_model(lr=1e-5)

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            callbacks=callbacks,
            class_weight=class_weight_dict
        )

    # ----------------------------
    # Evaluation
    # ----------------------------
    def evaluate_model(self):
        y_true = self.valid_generator.classes
        y_pred_prob = self.model.predict(self.valid_generator, verbose=1)
        y_pred = np.argmax(y_pred_prob, axis=1)

        cm = confusion_matrix(y_true, y_pred)

        print("\nCONFUSION MATRIX")
        print(cm)

        print("\nCLASSIFICATION REPORT")
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=list(self.valid_generator.class_indices.keys())
            )
        )

        f1 = f1_score(y_true, y_pred, average="weighted")
        print("Weighted F1 Score:", f1)

        