import shap
import numpy as np

from ..utils import MoleculeModelWrapper, binary_masker, CustomMultiHotAtomFeaturizer, CustomMultiHotBondFeaturizer
from rdkit import Chem
from api.ml_models import mpnn

from graphene import ObjectType, NonNull, List, String, Float

class AtomicQuery(ObjectType):
    interpret_permeability_by_atoms = List(Float, mol_smile=String(required=True))
    predict_permeability_by_atoms = Float(mol_smile=String(required=True))

    interpret_permeability_of_list_by_atoms = List(List(Float), mol_list=List(String, required=True))
    predict_permeability_of_list_by_atoms = List(Float, mol_list=List(String, required=True))

    def resolve_interpret_permeability_by_atoms(self, info, mol_smile):
        return helper_resolve_interpret_permeability_by_atoms(mol_smile)

    def resolve_predict_permeability_by_atoms(self, info, mol_smile):
        return helper_resolve_predict_permeability_by_atoms(mol_smile)
    
    def resolve_interpret_permeability_of_list_by_atoms(self, info, mol_list):
        return list(map(helper_resolve_interpret_permeability_by_atoms, mol_list))
    
    def resolve_predict_permeability_of_list_by_atoms(self, info, mol_list):
        return list(map(helper_resolve_predict_permeability_by_atoms, mol_list))
    
def helper_resolve_interpret_permeability_by_atoms(mol_smile):
    mol = Chem.MolFromSmiles(mol_smile)
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()

    atom_featurizer = CustomMultiHotAtomFeaturizer.v1()
    bond_featurizer = CustomMultiHotBondFeaturizer()

    model_wrapper = MoleculeModelWrapper(mol_smile, n_atoms, n_bonds, mpnn, atom_featurizer, bond_featurizer)
    explainer = shap.PermutationExplainer(model_wrapper, masker=binary_masker)

    keep_features = [1] * (n_atoms + n_bonds)
    feature_choice = np.array([keep_features])

    explanation = explainer(feature_choice)

    return explanation.values[0].tolist()

def helper_resolve_predict_permeability_by_atoms(mol_smile):
    mol = Chem.MolFromSmiles(mol_smile)
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()

    atom_featurizer = CustomMultiHotAtomFeaturizer.v1()
    bond_featurizer = CustomMultiHotBondFeaturizer()

    model_wrapper = MoleculeModelWrapper(mol_smile, n_atoms, n_bonds, mpnn, atom_featurizer, bond_featurizer)

    return model_wrapper.get_predictions(keep_atoms=None, keep_bonds=None)
           