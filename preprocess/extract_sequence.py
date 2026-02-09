import os
import time
import json
import pickle
import hydra
import warnings
from functools import partial
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from openbabel import pybel
from tqdm import tqdm
from joblib import Parallel, delayed
from omegaconf import DictConfig
from utils import molecule as mu
from utils.lmdb import read_lmdb
from utils.environment import PROJECT_ROOT


class ExtractSequence:
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess

        # Task specific settings
        self.task = 'gen'
        self.prop_name = process_cfg.mof_sequence.prop_name

        # Directories
        self.lmdb_dir = process_cfg.lmdb_dir
        self.seq_dir = process_cfg.seq_dir
        if not os.path.exists(self.seq_dir):
            os.makedirs(self.seq_dir)

        # Number of CPUs
        self.num_cpus = process_cfg.num_cpus  

    def process_one(self, idx, value):
        # Disable warnings within each process
        RDLogger.DisableLog('rdApp.*') 
        pybel.ob.obErrorLog.SetOutputLevel(0)
        warnings.filterwarnings("ignore")

        try:
            feats = pickle.loads(value)
            bb_mols = feats['bb_mols']

            metal_mols = []
            organic_mols = []

            for mol_str in bb_mols:
                bb_mol = Chem.MolFromMolBlock(mol_str, removeHs=True) # NOTE: remove hydrogens

                if mu.is_metal_bb(bb_mol):
                    metal_mols.append(bb_mol)
                else:
                    organic_mols.append(bb_mol)

            # Sort sequences by molecular weight
            metal_mols = sorted(metal_mols, key=lambda x: Descriptors.ExactMolWt(x))
            organic_mols = sorted(organic_mols, key=lambda x: Descriptors.ExactMolWt(x))

            # Convert to SMILES
            metal_smiles = [
                Chem.MolToSmiles(mol, canonical=True).replace(".", "")
                for mol in metal_mols
            ]
            organic_smiles = [
                Chem.MolToSmiles(mol, canonical=True)
                for mol in organic_mols
            ]

            # Construct string sequence
            metal_str = ".".join(metal_smiles)
            organic_str = ".".join(organic_smiles)
            mof_sequence = f"<BOS> {metal_str} <SEP> {organic_str} <EOS>"

            # Extract property
            prop_value = feats['prop_dict'].get(self.prop_name, None)

            return idx, (mof_sequence, prop_value)
            
        except Exception as e:
            print(f"Failed to process {idx}: {e}")
            return idx, None

    def process(self, split='train'):
        print(f"INFO:: Extracting sequences for {split} split...")

        # Start timer
        start_time = time.time()

        # Read data
        data_dict = {}
        src_env = read_lmdb(f"{self.lmdb_dir}/{self.task}/MetalOxo_final_{split}.lmdb")
        with src_env.begin() as src_txn:
            num_entries = src_env.stat()['entries']
            cursor = src_txn.cursor()
            for key_bytes, value in tqdm(cursor, desc="Reading data", total=num_entries):
                idx = int(key_bytes.decode('ascii'))
                data_dict[idx] = value
        src_env.close()

        # Process
        all_results = Parallel(n_jobs=self.num_cpus)(
            delayed(self.process_one)(idx, value)
            for idx, value in tqdm(data_dict.items(), desc="Processing data")
        )

        # Write as json
        result_dict = {}
        for idx, result in all_results:
            if result is not None:
                seq, prop = result
                if prop is not None:
                    result_dict[idx] = {
                        "seq": seq,
                        "prop": prop
                    }
        with open(f"{self.seq_dir}/mof_sequence_{split}.json", 'w') as f:
            json.dump(result_dict, f)

        print(f"INFO:: Wrote {len(result_dict)} entries to {self.seq_dir}/mof_sequence_{split}.json")
        print(f"INFO:: Time taken: {time.time() - start_time:.4f} s")


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="base.yaml")
def main(cfg: DictConfig):
    extractor = ExtractSequence(cfg=cfg)
    extractor.process(split="val")
    extractor.process(split="train")

if __name__ == "__main__":
    main()
