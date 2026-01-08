# https://github.com/nicola-decao/MolGAN/blob/master/utils/sparse_molecular_dataset.py

import pickle
import numpy as np

from rdkit import Chem
from rdkit.Chem import Crippen
from datetime import datetime


# LogP-based class definitions
LOGP_CLASS_HYDROPHILIC = 0  # LogP < 0
LOGP_CLASS_BALANCED = 1      # 0 <= LogP <= 2
NUM_LOGP_CLASSES = 2


class SparseMolecularDataset():

    def load(self, filename, subset=1, conditional=False):
        """
        Load dataset from file.
        
        Args:
            filename: Path to the .sparsedataset file
            subset: Fraction of data to use (0-1)
            conditional: If True, compute LogP labels and filter molecules with LogP > 2
        """
        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

        # Conditional generation: compute LogP and filter
        self.conditional = conditional
        self.num_classes = NUM_LOGP_CLASSES if conditional else 0
        
        if conditional:
            self._compute_logp_labels()
            self._filter_by_logp()

        self.train_idx = np.random.choice(self.train_idx, int(len(self.train_idx) * subset), replace=False)
        self.validation_idx = np.random.choice(self.validation_idx, int(len(self.validation_idx) * subset),
                                               replace=False)
        self.test_idx = np.random.choice(self.test_idx, int(len(self.test_idx) * subset), replace=False)

        self.train_count = len(self.train_idx)
        self.validation_count = len(self.validation_idx)
        self.test_count = len(self.test_idx)

        self.__len = self.train_count + self.validation_count + self.test_count
        
        if conditional:
            self._log_class_distribution()

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def _compute_logp_labels(self):
        """Compute LogP values and assign class labels for all molecules."""
        self.log('Computing LogP values and class labels...')
        
        data_logp = []
        data_class = []
        
        for mol in self.data:
            try:
                logp = Crippen.MolLogP(mol)
            except:
                logp = None
            
            data_logp.append(logp)
            
            # Assign class label
            if logp is None:
                label = -1  # Invalid, will be filtered
            elif logp < 0:
                label = LOGP_CLASS_HYDROPHILIC  # 0: Hydrophilic
            elif logp <= 2:
                label = LOGP_CLASS_BALANCED      # 1: Balanced
            else:
                label = -1  # LogP > 2, will be filtered
            
            data_class.append(label)
        
        self.data_logp = np.array(data_logp, dtype=np.float32)
        self.data_class = np.array(data_class, dtype=np.int32)
        
        self.log('Computed LogP for {} molecules'.format(len(self.data)))

    def _filter_by_logp(self):
        """Filter out molecules with LogP > 2 or invalid LogP."""
        self.log('Filtering molecules by LogP (keeping LogP <= 2)...')
        
        # Get indices of valid molecules (class 0 or 1)
        valid_mask = self.data_class >= 0
        valid_indices = np.where(valid_mask)[0]
        
        # Count before filtering
        total_before = len(self.data)
        hydrophilic_count = np.sum(self.data_class == LOGP_CLASS_HYDROPHILIC)
        balanced_count = np.sum(self.data_class == LOGP_CLASS_BALANCED)
        filtered_count = total_before - len(valid_indices)
        
        self.log('Before filtering: {} total, {} hydrophilic (LogP<0), {} balanced (0<=LogP<=2), {} excluded (LogP>2 or invalid)'.format(
            total_before, hydrophilic_count, balanced_count, filtered_count))
        
        # Filter all data arrays
        self.data = self.data[valid_indices]
        self.smiles = self.smiles[valid_indices]
        self.data_S = self.data_S[valid_indices]
        self.data_A = self.data_A[valid_indices]
        self.data_X = self.data_X[valid_indices]
        self.data_D = self.data_D[valid_indices]
        self.data_F = self.data_F[valid_indices]
        self.data_Le = self.data_Le[valid_indices]
        self.data_Lv = self.data_Lv[valid_indices]
        self.data_logp = self.data_logp[valid_indices]
        self.data_class = self.data_class[valid_indices]
        
        # Update indices - need to remap to new array positions
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
        
        # Filter and remap train/val/test indices
        self.train_idx = np.array([old_to_new[idx] for idx in self.train_idx if idx in old_to_new])
        self.validation_idx = np.array([old_to_new[idx] for idx in self.validation_idx if idx in old_to_new])
        self.test_idx = np.array([old_to_new[idx] for idx in self.test_idx if idx in old_to_new])
        
        # CRITICAL: Convert numpy arrays back to lists to fix 'in' operator for novelty/diversity checks
        # numpy fancy indexing returns numpy arrays, but the metrics functions expect lists
        if isinstance(self.data, np.ndarray):
            self.data = list(self.data)
        if isinstance(self.smiles, np.ndarray):
            self.smiles = list(self.smiles)
        
        # Create a set for O(1) novelty lookups (used by novel_scores)
        self.smiles_set = set(self.smiles)
        
        self.log('After filtering: {} molecules remaining'.format(len(self.data)))

    def _log_class_distribution(self):
        """Log the class distribution in train/val/test sets."""
        train_classes = self.data_class[self.train_idx]
        val_classes = self.data_class[self.validation_idx]
        test_classes = self.data_class[self.test_idx]
        
        self.log('Class distribution:')
        self.log('  Train: {} hydrophilic, {} balanced'.format(
            np.sum(train_classes == 0), np.sum(train_classes == 1)))
        self.log('  Val:   {} hydrophilic, {} balanced'.format(
            np.sum(val_classes == 0), np.sum(val_classes == 1)))
        self.log('  Test:  {} hydrophilic, {} balanced'.format(
            np.sum(test_classes == 0), np.sum(test_classes == 1)))
    
    def get_class_name(self, class_label):
        """Get human-readable class name."""
        if class_label == LOGP_CLASS_HYDROPHILIC:
            return 'hydrophilic'
        elif class_label == LOGP_CLASS_BALANCED:
            return 'balanced'
        else:
            return 'unknown'

    def generate(self, filename, add_h=False, filters=lambda x: True, size=None, validation=0.1, test=0.1):
        self.log('Extracting {}..'.format(filename))

        if filename.endswith('.sdf'):
            self.data = list(filter(lambda x: x is not None, Chem.SDMolSupplier(filename)))
        elif filename.endswith('.smi'):
            self.data = [Chem.MolFromSmiles(line) for line in open(filename, 'r').readlines()]

        self.data = list(map(Chem.AddHs, self.data)) if add_h else self.data
        self.data = list(filter(filters, self.data))
        self.data = self.data[:size]

        self.log('Extracted {} out of {} molecules {}adding Hydrogen!'.format(len(self.data),
                                                                              len(Chem.SDMolSupplier(filename)),
                                                                              '' if add_h else 'not '))

        self._generate_encoders_decoders()
        self._generate_AX()

        self.data = np.array(self.data)
        self.smiles = np.array(self.smiles)
        self.data_S = np.stack(self.data_S)
        self.data_A = np.stack(self.data_A)
        self.data_X = np.stack(self.data_X)
        self.data_D = np.stack(self.data_D)
        self.data_F = np.stack(self.data_F)
        self.data_Le = np.stack(self.data_Le)
        self.data_Lv = np.stack(self.data_Lv)

        self.vertexes = self.data_F.shape[-2]
        self.features = self.data_F.shape[-1]

        self._generate_train_validation_test(validation, test)

    def _generate_encoders_decoders(self):
        self.log('Creating atoms encoder and decoder..')
        atom_labels = sorted(set([atom.GetAtomicNum() for mol in self.data for atom in mol.GetAtoms()] + [0]))
        self.atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
        self.atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
        self.atom_num_types = len(atom_labels)
        self.log('Created atoms encoder and decoder with {} atom types and 1 PAD symbol!'.format(
            self.atom_num_types - 1))

        self.log('Creating bonds encoder and decoder..')
        bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType()
                                                                    for mol in self.data
                                                                    for bond in mol.GetBonds())))

        self.bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        self.bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        self.bond_num_types = len(bond_labels)
        self.log('Created bonds encoder and decoder with {} bond types and 1 PAD symbol!'.format(
            self.bond_num_types - 1))

        self.log('Creating SMILES encoder and decoder..')
        smiles_labels = ['E'] + list(set(c for mol in self.data for c in Chem.MolToSmiles(mol)))
        self.smiles_encoder_m = {l: i for i, l in enumerate(smiles_labels)}
        self.smiles_decoder_m = {i: l for i, l in enumerate(smiles_labels)}
        self.smiles_num_types = len(smiles_labels)
        self.log('Created SMILES encoder and decoder with {} types and 1 PAD symbol!'.format(
            self.smiles_num_types - 1))

    def _generate_AX(self):
        self.log('Creating features and adjacency matrices..')

        data = []
        smiles = []
        data_S = []
        data_A = []
        data_X = []
        data_D = []
        data_F = []
        data_Le = []
        data_Lv = []

        max_length = max(mol.GetNumAtoms() for mol in self.data)
        max_length_s = max(len(Chem.MolToSmiles(mol)) for mol in self.data)

        for i, mol in enumerate(self.data):
            A = self._genA(mol, connected=True, max_length=max_length)
            D = np.count_nonzero(A, -1)
            if A is not None:
                data.append(mol)
                smiles.append(Chem.MolToSmiles(mol))
                data_S.append(self._genS(mol, max_length=max_length_s))
                data_A.append(A)
                data_X.append(self._genX(mol, max_length=max_length))
                data_D.append(D)
                data_F.append(self._genF(mol, max_length=max_length))

                L = D - A
                Le, Lv = np.linalg.eigh(L)

                data_Le.append(Le)
                data_Lv.append(Lv)

        self.log(date=False)
        self.log('Created {} features and adjacency matrices  out of {} molecules!'.format(len(data),
                                                                                           len(self.data)))

        self.data = data
        self.smiles = smiles
        self.data_S = data_S
        self.data_A = data_A
        self.data_X = data_X
        self.data_D = data_D
        self.data_F = data_F
        self.data_Le = data_Le
        self.data_Lv = data_Lv
        self.__len = len(self.data)

    def _genA(self, mol, connected=True, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        A = np.zeros(shape=(max_length, max_length), dtype=np.int32)

        begin, end = [b.GetBeginAtomIdx() for b in mol.GetBonds()], [b.GetEndAtomIdx() for b in mol.GetBonds()]
        bond_type = [self.bond_encoder_m[b.GetBondType()] for b in mol.GetBonds()]

        A[begin, end] = bond_type
        A[end, begin] = bond_type

        degree = np.sum(A[:mol.GetNumAtoms(), :mol.GetNumAtoms()], axis=-1)

        return A if connected and (degree > 0).all() else None

    def _genX(self, mol, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        return np.array([self.atom_encoder_m[atom.GetAtomicNum()] for atom in mol.GetAtoms()] + [0] * (
                    max_length - mol.GetNumAtoms()), dtype=np.int32)

    def _genS(self, mol, max_length=None):

        max_length = max_length if max_length is not None else len(Chem.MolToSmiles(mol))

        return np.array([self.smiles_encoder_m[c] for c in Chem.MolToSmiles(mol)] + [self.smiles_encoder_m['E']] * (
                    max_length - len(Chem.MolToSmiles(mol))), dtype=np.int32)

    def _genF(self, mol, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        features = np.array([[*[a.GetDegree() == i for i in range(5)],
                              *[a.GetExplicitValence() == i for i in range(9)],
                              *[int(a.GetHybridization()) == i for i in range(1, 7)],
                              *[a.GetImplicitValence() == i for i in range(9)],
                              a.GetIsAromatic(),
                              a.GetNoImplicit(),
                              *[a.GetNumExplicitHs() == i for i in range(5)],
                              *[a.GetNumImplicitHs() == i for i in range(5)],
                              *[a.GetNumRadicalElectrons() == i for i in range(5)],
                              a.IsInRing(),
                              *[a.IsInRingSize(i) for i in range(2, 9)]] for a in mol.GetAtoms()], dtype=np.int32)

        return np.vstack((features, np.zeros((max_length - features.shape[0], features.shape[1]))))

    def matrices2mol(self, node_labels, edge_labels, strict=False):
        mol = Chem.RWMol()

        for node_label in node_labels:
            mol.AddAtom(Chem.Atom(self.atom_decoder_m[node_label]))

        for start, end in zip(*np.nonzero(edge_labels)):
            if start > end:
                mol.AddBond(int(start), int(end), self.bond_decoder_m[edge_labels[start, end]])

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol

    def seq2mol(self, seq, strict=False):
        mol = Chem.MolFromSmiles(''.join([self.smiles_decoder_m[e] for e in seq if e != 0]))

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol

    def _generate_train_validation_test(self, validation, test):

        self.log('Creating train, validation and test sets..')

        validation = int(validation * len(self))
        test = int(test * len(self))
        train = len(self) - validation - test

        self.all_idx = np.random.permutation(len(self))
        self.train_idx = self.all_idx[0:train]
        self.validation_idx = self.all_idx[train:train + validation]
        self.test_idx = self.all_idx[train + validation:]

        self.train_counter = 0
        self.validation_counter = 0
        self.test_counter = 0

        self.train_count = train
        self.validation_count = validation
        self.test_count = test

        self.log('Created train ({} items), validation ({} items) and test ({} items) sets!'.format(
            train, validation, test))

    def _next_batch(self, counter, count, idx, batch_size):
        if batch_size is not None:
            if counter + batch_size >= count:
                counter = 0
                np.random.shuffle(idx)

            batch_idx = idx[counter:counter + batch_size]
            output = [obj[batch_idx]
                      for obj in (self.data, self.smiles, self.data_S, self.data_A, self.data_X,
                                  self.data_D, self.data_F, self.data_Le, self.data_Lv)]
            
            # Add class labels if in conditional mode
            if self.conditional:
                output.append(self.data_class[batch_idx])

            counter += batch_size
        else:
            output = [obj[idx] for obj in (self.data, self.smiles, self.data_S, self.data_A, self.data_X,
                                           self.data_D, self.data_F, self.data_Le, self.data_Lv)]
            # Add class labels if in conditional mode
            if self.conditional:
                output.append(self.data_class[idx])

        return [counter] + output

    def next_train_batch(self, batch_size=None):
        """
        Get next training batch.
        
        Returns:
            If conditional=False: (mols, smiles, data_S, data_A, data_X, data_D, data_F, data_Le, data_Lv)
            If conditional=True:  (mols, smiles, data_S, data_A, data_X, data_D, data_F, data_Le, data_Lv, class_labels)
        """
        out = self._next_batch(counter=self.train_counter, count=self.train_count,
                               idx=self.train_idx, batch_size=batch_size)
        self.train_counter = out[0]

        return out[1:]

    def next_validation_batch(self, batch_size=None):
        """
        Get next validation batch.
        
        Returns:
            If conditional=False: (mols, smiles, data_S, data_A, data_X, data_D, data_F, data_Le, data_Lv)
            If conditional=True:  (mols, smiles, data_S, data_A, data_X, data_D, data_F, data_Le, data_Lv, class_labels)
        """
        out = self._next_batch(counter=self.validation_counter, count=self.validation_count,
                               idx=self.validation_idx, batch_size=batch_size)
        self.validation_counter = out[0]

        return out[1:]

    def next_test_batch(self, batch_size=None):
        """
        Get next test batch.
        
        Returns:
            If conditional=False: (mols, smiles, data_S, data_A, data_X, data_D, data_F, data_Le, data_Lv)
            If conditional=True:  (mols, smiles, data_S, data_A, data_X, data_D, data_F, data_Le, data_Lv, class_labels)
        """
        out = self._next_batch(counter=self.test_counter, count=self.test_count,
                               idx=self.test_idx, batch_size=batch_size)
        self.test_counter = out[0]

        return out[1:]
    
    def sample_class_labels(self, batch_size):
        """
        Sample random class labels for generator training.
        
        Args:
            batch_size: Number of labels to sample
            
        Returns:
            numpy array of shape (batch_size,) with class labels (0 or 1)
        """
        return np.random.randint(0, self.num_classes, size=batch_size)

    @staticmethod
    def log(msg='', date=True):
        print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ' ' + str(msg) if date else str(msg))

    def __len__(self):
        return self.__len


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='generate', choices=['generate', 'analyze'])
    parser.add_argument('--input', type=str, default='gdb9.sdf')
    parser.add_argument('--output', type=str, default='gdb9_9nodes.sparsedataset')
    args = parser.parse_args()
    
    if args.mode == 'generate':
        # Generate dataset from SDF file
        data = SparseMolecularDataset()
        data.generate(args.input, filters=lambda x: x.GetNumAtoms() <= 9)
        data.save(args.output)
        
    elif args.mode == 'analyze':
        # Analyze LogP distribution in existing dataset
        data = SparseMolecularDataset()
        data.load(args.output, conditional=True)
        
        print('\n=== LogP Distribution Analysis ===')
        print('Total molecules: {}'.format(len(data.data)))
        print('Hydrophilic (LogP < 0): {}'.format(np.sum(data.data_class == 0)))
        print('Balanced (0 <= LogP <= 2): {}'.format(np.sum(data.data_class == 1)))
        print('LogP range: [{:.2f}, {:.2f}]'.format(data.data_logp.min(), data.data_logp.max()))
        print('LogP mean: {:.2f}, std: {:.2f}'.format(data.data_logp.mean(), data.data_logp.std()))
