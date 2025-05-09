import xgboost
from os import path
from chemprop import models

optimal_xgb_descriptors = xgboost.XGBRegressor()
optimal_xgb_descriptors.load_model(path.abspath("api/ml_models/optimal_xgb_descriptors.bin"))

mpnn = models.MPNN.load_from_file(path.abspath("api/ml_models/target_inverted_model_v2.pt"))