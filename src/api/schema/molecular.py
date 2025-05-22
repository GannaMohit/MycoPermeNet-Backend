from graphene import Float, List, ObjectType, Scalar, NonNull, Int, String
import shap
import pandas as pd

from os import path

from sklearn.neighbors import NearestNeighbors

from api.ml_models import optimal_xgb_descriptors

COLUMNS = ['HBA', 'HBD', 'HBA+HBD', 'NumRings', 'RTB', 'NumAmideBonds',
               'Globularity', 'PBF', 'TPSA', 'logP', 'MR', 'MW', 'Csp3',
               'fmf', 'QED', 'HAC', 'NumRingsFused', 'unique_HBAD', 'max_ring_size',
               'n_chiral_centers', 'fcsp3_bm', 'formal_charge', 'abs_charge']

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

class SimilarMolecule(Scalar):
    smile = NonNull(String)
    distance = NonNull(Float)

    
class MolecularQuery(ObjectType):
    interpret_permeability_by_molecular_descriptors = List(Float, descriptors=Descriptors(required=True))
    predict_permeability_by_molecular_descriptors = Float(descriptors=Descriptors(required=True))

    find_similar_molecules_by_molecular_descriptors = List(SimilarMolecule, descriptors=Descriptors(required=True))
    
    def resolve_interpret_permeability_by_molecular_descriptors(self, info, descriptors):
        all_descriptors = pd.read_csv(path.abspath("api/data/all_descriptors.csv"))
        explainer = shap.Explainer(optimal_xgb_descriptors, all_descriptors[COLUMNS])
        explanation = explainer(pd.DataFrame([descriptors], columns=COLUMNS))
        return explanation.values[0].tolist()
    
    def resolve_predict_permeability_by_molecular_descriptors(self, info, descriptors):
        return optimal_xgb_descriptors.predict(pd.DataFrame([descriptors], columns=COLUMNS))
    
    def resolve_find_similar_molecules_by_molecular_descriptors(self, info, descriptors):
        data = pd.read_csv(path.abspath("api/data/all_molecule_descriptor_dataset.csv"), header=0, index_col=0)
       
        nn = NearestNeighbors()
        nn.fit(data[COLUMNS].dropna())

        # Find nearest neighbors
        distances, indices = nn.kneighbors(pd.DataFrame([descriptors], columns=COLUMNS), n_neighbors=10)

        similar_rows = data.iloc[indices[0]]['SMILES']

        return list(zip(similar_rows.to_list(), distances[0].tolist()))
