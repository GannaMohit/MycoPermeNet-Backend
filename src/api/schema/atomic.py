import graphene
import shap
import numpy as np
from os import path

from ..utils import MoleculeModelWrapper, binary_masker, CustomMultiHotAtomFeaturizer, CustomMultiHotBondFeaturizer
from rdkit import Chem
from chemprop import models

class AtomicQuery(graphene.ObjectType):
    interpret_permeability_by_atoms = graphene.List(graphene.Float, mol_smile=graphene.String(required=True))

    def resolve_interpret_permeability_by_atoms(self, info, mol_smile):
        mol = Chem.MolFromSmiles(mol_smile)
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        mpnn = models.MPNN.load_from_file(path.abspath("api/ml_models/target_inverted_model_v2.pt"))

        atom_featurizer = CustomMultiHotAtomFeaturizer.v1()
        bond_featurizer = CustomMultiHotBondFeaturizer()

        model_wrapper = MoleculeModelWrapper(mol_smile, n_atoms, n_bonds, mpnn, atom_featurizer, bond_featurizer)
        explainer = shap.PermutationExplainer(model_wrapper, masker=binary_masker)

        keep_features = [1] * (n_atoms + n_bonds)
        feature_choice = np.array([keep_features])

        explanation = explainer(feature_choice)

        return explanation.values[0].tolist()
    
# schema = graphene.Schema(query=AtomicQuery)