from django.apps import AppConfig


class DetectorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "detector"

    def ready(self):
        # Load model/vectorizer/scaler once when the server starts, not on
        # every request -- joblib.load() and NLTK stopword lookup aren't
        # free, and a live demo should respond instantly per request.
        from . import ml
        ml.load_artifacts()
