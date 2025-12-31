# import os
# import numpy as np
# import pandas as pd
# import pickle
# import logging
# import yaml

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

# # --------------------------------------------------
# # LOGGING SETUP
# # --------------------------------------------------
# log_dir = "logs"
# os.makedirs(log_dir, exist_ok=True)

# logger = logging.getLogger("model_building")
# logger.setLevel(logging.DEBUG)

# console_handler = logging.StreamHandler()
# file_handler = logging.FileHandler(os.path.join(log_dir, "model_building.log"))

# formatter = logging.Formatter(
#     "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )

# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# logger.addHandler(console_handler)
# logger.addHandler(file_handler)

# # --------------------------------------------------
# # UTIL FUNCTIONS
# # --------------------------------------------------
# def load_params(params_path: str) -> dict:
#     try:
#         with open(params_path, "r") as file:
#             params = yaml.safe_load(file)
#         logger.debug("Parameters retrieved from %s", params_path)
#         return params
#     except Exception as e:
#         logger.error("Failed to load params.yaml: %s", e)
#         raise


# def load_data(file_path: str) -> pd.DataFrame:
#     try:
#         df = pd.read_csv(file_path)
#         logger.debug("Data loaded from %s with shape %s", file_path, df.shape)
#         return df
#     except Exception as e:
#         logger.error("Failed to load data: %s", e)
#         raise

# # --------------------------------------------------
# # MODEL FACTORY (CORE FIX)
# # --------------------------------------------------
# def get_model(model_name: str, model_params: dict, random_state: int):
#     logger.debug(
#         "Initializing %s model with parameters: %s",
#         model_name,
#         model_params
#     )

#     if model_name == "random_forest":
#         return RandomForestClassifier(
#             random_state=random_state,
#             **model_params
#         )

#     elif model_name == "logistic_regression":
#         return LogisticRegression(
#             random_state=random_state,
#             **model_params
#         )

#     else:
#         raise ValueError(f"Unsupported model: {model_name}")

# # --------------------------------------------------
# # TRAINING
# # --------------------------------------------------
# def train_model(X_train: np.ndarray, y_train: np.ndarray, model):
#     try:
#         if X_train.shape[0] != y_train.shape[0]:
#             raise ValueError("X and y sample size mismatch")

#         logger.debug("Model training started with %d samples", X_train.shape[0])
#         model.fit(X_train, y_train)
#         logger.debug("Model training completed")

#         return model

#     except Exception as e:
#         logger.error("Error during model training: %s", e)
#         raise

# # --------------------------------------------------
# # SAVE MODEL
# # --------------------------------------------------
# def save_model(model, file_path: str):
#     try:
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         with open(file_path, "wb") as f:
#             pickle.dump(model, f)
#         logger.debug("Model saved to %s", file_path)
#     except Exception as e:
#         logger.error("Failed to save model: %s", e)
#         raise

# # --------------------------------------------------
# # MAIN PIPELINE
# # --------------------------------------------------
# def main():
#     try:
#         # ---- Load config ----
#         all_params = load_params("params.yaml")

#         model_cfg = all_params["model_building"]
#         models_cfg = all_params["models"]

#         active_model = model_cfg["active_model"]
#         random_state = model_cfg.get("random_state", 42)

#         model_params = models_cfg[active_model]

#         # ---- Load data ----
#         train_data = load_data("./data/processed/train_tfidf.csv")
#         X_train = train_data.iloc[:, :-1].values
#         y_train = train_data.iloc[:, -1].values

#         # ---- Build model ----
#         model = get_model(
#             model_name=active_model,
#             model_params=model_params,
#             random_state=random_state
#         )

#         # ---- Train ----
#         trained_model = train_model(X_train, y_train, model)

#         # ---- Save ----
#         save_model(trained_model, "models/model.pkl")

#         logger.info("Model training pipeline completed successfully")

#     except Exception as e:
#         logger.error("Pipeline failed: %s", e)
#         print(f"Error: {e}")

# # --------------------------------------------------
# if __name__ == "__main__":
#     main()


import os
import numpy as np
import pandas as pd
import pickle
import logging
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# LOGGING SETUP (SAFE)
# --------------------------------------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, "model_building.log"))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# --------------------------------------------------
# UTIL FUNCTIONS
# --------------------------------------------------
def load_params(params_path="params.yaml") -> dict:
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"{params_path} not found")

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    logger.debug("Parameters loaded from %s", params_path)
    return params


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    df = pd.read_csv(file_path)
    logger.debug("Data loaded from %s with shape %s", file_path, df.shape)
    return df

# --------------------------------------------------
# MODEL FACTORY
# --------------------------------------------------
def get_model(model_name: str, model_params: dict, random_state: int):
    logger.debug(
        "Initializing model '%s' with params %s",
        model_name,
        model_params
    )

    if model_name == "random_forest":
        return RandomForestClassifier(
            random_state=random_state,
            **model_params
        )

    elif model_name == "logistic_regression":
        return LogisticRegression(
            random_state=random_state,
            **model_params
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

# --------------------------------------------------
# TRAINING
# --------------------------------------------------
def train_model(X: np.ndarray, y: np.ndarray, model):
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y size mismatch")

    logger.debug("Training started on %d samples", X.shape[0])
    model.fit(X, y)
    logger.debug("Training completed")
    return model

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------
def save_model(model, path="models/model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)

    logger.debug("Model saved to %s", path)

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def main():
    try:
        params = load_params()

        if "model_building" not in params:
            raise KeyError("Missing 'model_building' in params.yaml")

        if "models" not in params:
            raise KeyError("Missing 'models' in params.yaml")

        model_cfg = params["model_building"]
        models_cfg = params["models"]

        active_model = model_cfg["active_model"]
        random_state = model_cfg.get("random_state", 42)

        if active_model not in models_cfg:
            raise KeyError(f"Model '{active_model}' not found in params.yaml")

        model_params = models_cfg[active_model]

        # Load data
        df = load_data("data/processed/train_tfidf.csv")
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Build + Train
        model = get_model(active_model, model_params, random_state)
        trained_model = train_model(X, y, model)

        # Save
        save_model(trained_model)

        logger.info("Model building pipeline completed successfully")

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()
