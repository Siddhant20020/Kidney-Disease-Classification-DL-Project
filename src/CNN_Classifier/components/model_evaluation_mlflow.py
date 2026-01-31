import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from CNN_Classifier.entity.config_entity import EvaluationConfig
from CNN_Classifier.utils.common import read_yaml, create_directories,save_json
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)


class Evaluation:
    def __init__(self, config):
        self.config = config

    # ----------------------------
    # Validation Generator
    # ----------------------------
    def _valid_generator(self):
        datagenerator_kwargs = dict(
            preprocessing_function=preprocess_input,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            shuffle=False
        )

        valid_datagenerator = ImageDataGenerator(**datagenerator_kwargs)

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            **dataflow_kwargs
        )

    # ----------------------------
    # Load Model
    # ----------------------------
    def load_model(self):
        self.model = tf.keras.models.load_model(
            self.config.path_of_model,
            compile=False  
        )

        # Recompile SAFELY for evaluation
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    # ----------------------------
    # Evaluation
    # ----------------------------
    def evaluation(self):
        self.load_model()
        self._valid_generator()

        # ---- Keras evaluation ----
        self.score = self.model.evaluate(
            self.valid_generator,
            verbose=1
        )

        # ---- Predictions ----
        y_true = self.valid_generator.classes
        y_pred_prob = self.model.predict(self.valid_generator, verbose=1)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # ---- Metrics (SKLEARN) ----
        self.f1 = f1_score(y_true, y_pred, average="weighted")
        self.precision = precision_score(y_true, y_pred, average="weighted")
        self.recall = recall_score(y_true, y_pred, average="weighted")

        self.report = classification_report(
            y_true,
            y_pred,
            target_names=list(self.valid_generator.class_indices.keys())
        )

        self.cm = confusion_matrix(y_true, y_pred)

        print("\nClassification Report:\n", self.report)
        print("Confusion Matrix:\n", self.cm)

    # ----------------------------
    # Save Scores
    # ----------------------------
    def save_score(self):
        scores = {
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1]),
            "f1_score": float(self.f1),
            "precision": float(self.precision),
            "recall": float(self.recall)
        }

        save_json(path=Path("scores.json"), data=scores)

    # ----------------------------
    # MLflow Logging
    # ----------------------------
    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(
            mlflow.get_tracking_uri()
        ).scheme

        with mlflow.start_run(run_name="ResNet50_4Class_Eval"):
            mlflow.log_params(self.config.all_params)

            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1],
                "f1_score": self.f1,
                "precision": self.precision,
                "recall": self.recall
            })

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model,
                    "model",
                    registered_model_name="ResNet50_Kidney_4Class"
                )
            else:
                mlflow.keras.log_model(self.model, "model")