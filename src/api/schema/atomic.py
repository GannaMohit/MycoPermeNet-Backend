import graphene
import shap
import numpy as np

from ..utils import MoleculeModelWrapper, binary_masker, CustomMultiHotAtomFeaturizer, CustomMultiHotBondFeaturizer
from rdkit import Chem
from api.ml_models import mpnn, mlp_optimal

class AtomicQuery(graphene.ObjectType):
    interpret_permeability_by_atoms = graphene.List(graphene.Float, mol_smile=graphene.String(required=True))
    predict_permeability_by_smile = graphene.Float(mol_smile=graphene.String(required=True))

    def resolve_interpret_permeability_by_atoms(self, info, mol_smile):
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
    
    # TODO: resolve_predict_permeability_by_smile