from os import path
from graphene import Enum, Int, ObjectType, NonNull, List, Scalar, String, Float

from rdkit.Chem import MolFromSmiles, MACCSkeys, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetRDKitFPGenerator, GetTopologicalTorsionGenerator
from rdkit.Avalon.pyAvalonTools import GetAvalonFP

import pickle
import pandas as pd
import gzip

class TanimotoFingerprintScores(Scalar):
    smiles = NonNull(String)
    maccs = NonNull(Float)
    avalon = NonNull(Float)
    morgan = NonNull(Float)
    atom_pair = NonNull(Float)
    topological = NonNull(Float)
    rdkit = NonNull(Float)

class Fingerprints(Enum):
    maccs = "maccs"
    avalon = "avalon"
    morgan = "morgan"
    atom_pair = "atom_pair"
    topological = "topological"
    rdkit = "rdkit"

class SimilarityQuery(ObjectType):
    find_similar_molecules_by_tanimoto_similarity = List(TanimotoFingerprintScores, mol_smiles = String(required=True), max_output_length=Int(default_value=1000), sorting_fingerprint=Fingerprints(default_value='maccs'))

    def resolve_find_similar_molecules_by_tanimoto_similarity(self, info, mol_smiles, max_output_length=1000, sorting_fingerprint='maccs'):
        with gzip.open(path.abspath("api/data/fingerprints.pkl.gz"), 'rb') as file:
            fingerprints = pickle.load(file)
        
        mol = MolFromSmiles(mol_smiles)
        
        morgan_generator = GetMorganGenerator()
        atom_pair_generator = GetAtomPairGenerator()
        topological_generator = GetTopologicalTorsionGenerator()
        rdkitfp_generator = GetRDKitFPGenerator()

        fps = {
            'maccs': MACCSkeys.GenMACCSKeys(mol),
            'avalon': GetAvalonFP(mol),
            'morgan': morgan_generator.GetFingerprint(mol),
            'atom_pair': atom_pair_generator.GetFingerprint(mol),
            'topological': topological_generator.GetFingerprint(mol),
            'rdkit': rdkitfp_generator.GetFingerprint(mol)
        }

        df = pd.DataFrame(map(lambda x: x['smiles'], fingerprints), columns=['smiles'])

        for key in ['maccs', 'avalon', 'morgan', 'atom_pair', 'topological', 'rdkit']:
            df[key] = DataStructs.BulkTanimotoSimilarity(fps[key], list(map(lambda x: x[key], fingerprints)))

        return df.nlargest(max_output_length, columns=sorting_fingerprint).values.tolist()