from os import path
from graphene import Float, List, ObjectType, Scalar, NonNull, Int, String
import shap
import xgboost
import pandas as pd

class Descriptors(Scalar):
    hba = NonNull(Int)
    hbd = NonNull(Int)
    hba_plus_hbd = NonNull(Int)
    nrings = NonNull(Int)
    rtb = NonNull(Int)
    n_amide_bond = NonNull(Int)
    glob = NonNull(Float)
    pbf = NonNull(Float)
    psa = NonNull(Float)
    logp = NonNull(Float)
    mr = NonNull(Float)
    mw = NonNull(Float)
    csp3 = NonNull(Float)
    fmf = NonNull(Float)
    qed = NonNull(Float)
    hac = NonNull(Int)
    nrings_fused = NonNull(Int)
    n_unique_hba_hbd_atoms = NonNull(Int)
    max_ring_size = NonNull(Int)
    n_chiral_centers = NonNull(Int)
    fcsp3_bm = NonNull(Float)
    f_charge = NonNull(Int)
    abs_charge = NonNull(Int)
    
class MolecularQuery(ObjectType):
    interpret_molecular = List(Float, descriptors=Descriptors(required=True))

    def resolve_interpret_molecular(self, info, descriptors):
        optimal_xgb_descriptors = xgboost.XGBRegressor()
        optimal_xgb_descriptors.load_model(path.abspath("api/ml_models/optimal_xgb_descriptors.bin"))

        all_descriptors = pd.read_csv(path.abspath("api/data/all_descriptors.csv"))
        
        explainer = shap.Explainer(optimal_xgb_descriptors, all_descriptors[all_descriptors.columns[1:]])

        explanation = explainer(pd.DataFrame([descriptors], columns=all_descriptors.columns[1:]))

        print(explanation)

        return explanation.values[0].tolist()