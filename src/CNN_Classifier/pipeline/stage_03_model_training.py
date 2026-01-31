from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.model_training import Training
from CNN_Classifier import logger
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)

STAGE_NAME="Training"

class ModelTrainingPipeline:
  def __init__(self):
    pass
  
  def main(self):
      config = ConfigurationManager()
      training_config = config.get_training_config()
      training = Training(config=training_config)
      training.get_base_model()
      training.train_valid_generator()
      training.train()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        pipeline = ModelTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e