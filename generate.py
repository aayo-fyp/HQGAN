#!/usr/bin/env python3
"""
Conditional Molecular Generation Script.

Generate molecules of a specific LogP class:
- Class 0 (hydrophilic): LogP < 0
- Class 1 (balanced): 0 <= LogP <= 2

Usage:
    python generate.py --class_type hydrophilic --num_samples 100 --model_dir results/conditional-GAN/...
    python generate.py --class_type balanced --num_samples 50
"""

import os
import argparse
import numpy as np
import torch
import pennylane as qml
import datetime
import json

from rdkit import Chem
from rdkit.Chem import Crippen, AllChem, Draw

from models.models import ConditionalGenerator
from data.sparse_molecular_dataset import SparseMolecularDataset, LOGP_CLASS_HYDROPHILIC, LOGP_CLASS_BALANCED
from utils.utils import save_mol_img, MolecularMetrics, all_scores
from utils.utils_io import get_date_postfix
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description='Generate molecules with specific LogP class')
    
    parser.add_argument('--class_type', type=str, required=True, 
                        choices=['hydrophilic', 'balanced', '0', '1'],
                        help='Target class: hydrophilic (LogP<0) or balanced (0<=LogP<=2)')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of molecules to generate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for generation (default: 16, same as training)')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model checkpoints')
    parser.add_argument('--model_epoch', type=int, default=None,
                        help='Epoch of model to load (default: latest)')
    parser.add_argument('--output_base', type=str, default='results/generated',
                        help='Base output directory (default: results/generated)')
    
    # Model config
    parser.add_argument('--z_dim', type=int, default=8, help='Noise dimension')
    parser.add_argument('--class_embed_dim', type=int, default=8, help='Class embedding dimension')
    parser.add_argument('--complexity', type=str, default='mr', choices=['nr', 'mr', 'hr'])
    
    # Quantum config
    parser.add_argument('--quantum', action='store_true', help='Use quantum noise generation')
    parser.add_argument('--qubits', type=int, default=8, help='Number of qubits')
    parser.add_argument('--layer', type=int, default=3, help='Number of quantum layers')
    
    # Dataset for getting encoders/decoders
    parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes.sparsedataset',
                        help='Path to molecular dataset')
    
    return parser.parse_args()


def setup_output_directories(args, class_name):
    """
    Create standardized output directory structure.
    
    Structure:
        results/generated/
        └── YYYYMMDD_HHMMSS_<class_name>/
            ├── img_dir/          # Individual molecule images
            ├── grid_dir/         # Grid summary images  
            ├── smiles/           # SMILES files
            └── config.json       # Generation config for reproducibility
    """
    timestamp = get_date_postfix()
    run_name = f"{timestamp}_{class_name}"
    
    # Create directory structure
    run_dir = os.path.join(args.output_base, run_name)
    img_dir = os.path.join(run_dir, 'img_dir')
    grid_dir = os.path.join(run_dir, 'grid_dir')
    smiles_dir = os.path.join(run_dir, 'smiles')
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(smiles_dir, exist_ok=True)
    
    return {
        'run_dir': run_dir,
        'img_dir': img_dir,
        'grid_dir': grid_dir,
        'smiles_dir': smiles_dir,
        'timestamp': timestamp,
        'run_name': run_name
    }


def save_generation_config(dirs, args, class_name, class_desc, model_epoch, stats):
    """Save generation configuration for reproducibility"""
    config = {
        'timestamp': dirs['timestamp'],
        'class_type': class_name,
        'class_description': class_desc,
        'num_samples': args.num_samples,
        'model_dir': args.model_dir,
        'model_epoch': model_epoch,
        'z_dim': args.z_dim,
        'class_embed_dim': args.class_embed_dim,
        'complexity': args.complexity,
        'quantum': args.quantum,
        'qubits': args.qubits if args.quantum else None,
        'layer': args.layer if args.quantum else None,
        'results': {
            'total_generated': stats['total'],
            'valid_molecules': stats['valid'],
            'validity_rate': stats['validity_rate'],
            'class_accuracy': stats['class_accuracy'],
            'molecular_quality': {
                'validity': stats.get('valid', 0.0),
                'uniqueness': stats.get('unique', 0.0),
                'novelty': stats.get('novel', 0.0)
            },
            'drug_likeness': {
                'qed_mean': stats.get('qed_mean', 0.0),
                'sa_score_mean': stats.get('sa_mean', 0.0),
                'np_score_mean': stats.get('np_mean', 0.0),
                'diversity_mean': stats.get('diversity_mean', 0.0),
                'drug_candidate_mean': stats.get('drugcand_mean', 0.0)
            },
            'logp_raw': {
                'mean': stats['logp_mean'],
                'std': stats['logp_std'],
                'min': stats['logp_min'],
                'max': stats['logp_max']
            },
            'logp_normalized': {
                'mean': stats['logp_norm_mean'],
                'std': stats['logp_norm_std'],
                'note': 'Normalized to [0,1] from range [-2.12, 6.04]'
            }
        }
    }
    
    config_path = os.path.join(dirs['run_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def get_generator_dims(complexity):
    """Get generator conv dimensions based on complexity"""
    if complexity == 'nr':
        return [128, 256, 512]
    elif complexity == 'mr':
        return [128]
    elif complexity == 'hr':
        return [16]
    else:
        raise ValueError(f"Unknown complexity: {complexity}")


def find_latest_checkpoint(model_dir):
    """Find the latest model checkpoint in directory"""
    g_files = [f for f in os.listdir(model_dir) if f.endswith('-G.ckpt')]
    if not g_files:
        raise FileNotFoundError(f"No generator checkpoints found in {model_dir}")
    
    # Extract epoch numbers
    epochs = [int(f.split('-')[0]) for f in g_files]
    latest_epoch = max(epochs)
    return latest_epoch


def load_generator(args, data):
    """Load trained conditional generator"""
    
    # Get model dimensions
    g_conv_dim = get_generator_dims(args.complexity)
    
    # Create generator
    G = ConditionalGenerator(
        conv_dims=g_conv_dim,
        z_dim=args.z_dim,
        vertexes=data.vertexes,
        edges=data.bond_num_types,
        nodes=data.atom_num_types,
        dropout_rate=0.0,  # No dropout during inference
        num_classes=2,
        class_embed_dim=args.class_embed_dim
    )
    
    # Find checkpoint
    if args.model_epoch is None:
        epoch = find_latest_checkpoint(args.model_dir)
    else:
        epoch = args.model_epoch
    
    # Load weights
    G_path = os.path.join(args.model_dir, f'{epoch}-G.ckpt')
    print(f'Loading generator from {G_path}')
    G.load_state_dict(torch.load(G_path, map_location='cpu'))
    G.eval()
    
    return G, epoch


def create_quantum_circuit(qubits, layer):
    """Create quantum circuit for noise generation"""
    dev = qml.device('default.qubit', wires=qubits)
    
    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def gen_circuit(w):
        import random
        z1 = random.uniform(-1, 1)
        z2 = random.uniform(-1, 1)
        for i in range(qubits):
            qml.RY(np.arcsin(z1), wires=i)
            qml.RZ(np.arcsin(z2), wires=i)
        for l in range(layer):
            for i in range(qubits):
                qml.RY(w[i], wires=i)
            for i in range(qubits-1):
                qml.CNOT(wires=[i, i+1])
                qml.RZ(w[i+qubits], wires=i+1)
                qml.CNOT(wires=[i, i+1])
        return [qml.expval(qml.PauliZ(i)) for i in range(qubits)]
    
    return gen_circuit


def postprocess_stochastic(edges_logits, nodes_logits, method='soft_gumbel', temperature=1.0):
    """Postprocess logits using stochastic sampling (soft_gumbel) for diversity"""
    if method == 'soft_gumbel':
        # Stochastic Gumbel-Softmax (soft) - more diverse
        edges_soft = F.gumbel_softmax(edges_logits.contiguous().view(-1, edges_logits.size(-1))/temperature, hard=False).view(edges_logits.size())
        nodes_soft = F.gumbel_softmax(nodes_logits.contiguous().view(-1, nodes_logits.size(-1))/temperature, hard=False).view(nodes_logits.size())
        edges_hard = torch.argmax(edges_soft, dim=-1)
        nodes_hard = torch.argmax(nodes_soft, dim=-1)
    elif method == 'hard_gumbel':
        # Hard Gumbel-Softmax - still stochastic but discrete
        edges_hard = F.gumbel_softmax(edges_logits.contiguous().view(-1, edges_logits.size(-1))/temperature, hard=True).view(edges_logits.size())
        nodes_hard = F.gumbel_softmax(nodes_logits.contiguous().view(-1, nodes_logits.size(-1))/temperature, hard=True).view(nodes_logits.size())
        edges_hard = torch.argmax(edges_hard, dim=-1)
        nodes_hard = torch.argmax(nodes_hard, dim=-1)
    else:
        # Fallback to softmax + argmax
        edges_hard = torch.argmax(torch.softmax(edges_logits, dim=-1), dim=-1)
        nodes_hard = torch.argmax(torch.softmax(nodes_logits, dim=-1), dim=-1)
    
    return edges_hard, nodes_hard


def generate_molecules(G, data, target_class, num_samples, args, device='cpu', batch_size=16):
    """Generate molecules of specified class in batches for better diversity"""
    
    all_mols = []
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    print(f'Generating {num_samples} molecules in {num_batches} batches of {batch_size}...')
    
    # Setup quantum circuit if needed
    if args.quantum:
        print('Using quantum noise generation...')
        gen_circuit = create_quantum_circuit(args.qubits, args.layer)
        
        # Load quantum weights if available
        qc_weights_path = os.path.join(args.model_dir, 'molgan_red_weights.csv')
        if os.path.exists(qc_weights_path):
            import pandas as pd
            weights = pd.read_csv(qc_weights_path, header=None).iloc[-1, 1:].values
            gen_weights = torch.tensor(list(weights), requires_grad=False)
        else:
            gen_weights = torch.tensor(list(np.random.rand(args.layer*(args.qubits*2-1))*2*np.pi-np.pi), requires_grad=False)
    
    # Generate in batches
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - len(all_mols))
        
        # Create target labels for this batch
        target_labels = torch.full((current_batch_size,), target_class, dtype=torch.long).to(device)
        
        # Generate noise for this batch
        if args.quantum:
            z = torch.stack([torch.stack(gen_circuit(gen_weights)) for _ in range(current_batch_size)]).to(device).float()
        else:
            z = torch.randn(current_batch_size, args.z_dim).to(device)
        
        # Generate molecules
        with torch.no_grad():
            edges_logits, nodes_logits = G(z, target_labels)
            
            # Use stochastic sampling (soft_gumbel) for diversity
            edges_hard, nodes_hard = postprocess_stochastic(edges_logits, nodes_logits, method='soft_gumbel', temperature=1.0)
        
        # Convert to molecules
        for i in range(current_batch_size):
            mol = data.matrices2mol(
                nodes_hard[i].cpu().numpy(),
                edges_hard[i].cpu().numpy(),
                strict=True
            )
            all_mols.append(mol)
        
        if (batch_idx + 1) % 10 == 0:
            print(f'  Generated batch {batch_idx + 1}/{num_batches} ({len(all_mols)}/{num_samples} molecules)')
    
    print(f'Generated {len(all_mols)} molecules total')
    return all_mols


def compute_logp_stats(mols, target_class):
    """Compute LogP statistics for generated molecules (both raw and normalized)"""
    valid_mols = []
    logp_values = []  # LogP for valid molecules only
    logp_all = []     # LogP for all molecules (None for invalid)
    correct_class = 0
    
    for mol in mols:
        if mol is not None:
            try:
                logp = Crippen.MolLogP(mol)
                valid_mols.append(mol)
                logp_values.append(logp)
                logp_all.append(logp)
                
                # Check class (using RAW LogP thresholds)
                if target_class == 0 and logp < 0:
                    correct_class += 1
                elif target_class == 1 and 0 <= logp <= 2:
                    correct_class += 1
            except:
                logp_all.append(None)
        else:
            logp_all.append(None)
    
    # Compute normalized LogP: maps from [-2.12, 6.04] to [0, 1]
    logp_norm = []
    if logp_values:
        logp_raw = np.array(logp_values)
        logp_norm = np.clip((logp_raw - (-2.12178879609)) / (6.0429063424 - (-2.12178879609)), 0.0, 1.0)
    
    return {
        'total': len(mols),
        'valid': len(valid_mols),
        'validity_rate': len(valid_mols) / len(mols) * 100 if mols else 0,
        'logp_values': logp_values,
        'logp_all': logp_all,  # Includes None for invalid molecules
        # Raw LogP statistics
        'logp_mean': np.mean(logp_values) if logp_values else 0,
        'logp_std': np.std(logp_values) if logp_values else 0,
        'logp_min': np.min(logp_values) if logp_values else 0,
        'logp_max': np.max(logp_values) if logp_values else 0,
        # Normalized LogP statistics [0, 1]
        'logp_norm_mean': np.mean(logp_norm) if len(logp_norm) > 0 else 0,
        'logp_norm_std': np.std(logp_norm) if len(logp_norm) > 0 else 0,
        # Class accuracy
        'class_accuracy': correct_class / len(valid_mols) * 100 if valid_mols else 0,
        'correct_class': correct_class,
        'valid_mols': valid_mols
    }


def compute_all_metrics(mols, data):
    """Compute all molecular metrics: validity, uniqueness, novelty, QED, SA, NP, diversity, drug candidate"""
    try:
        # Compute all scores using the utility function
        m0, m1 = all_scores(mols, data, norm=False, reconstruction=False)
        
        # Extract metrics
        metrics = {
            'valid': m1.get('valid', 0.0),
            'unique': m1.get('unique', 0.0),
            'novel': m1.get('novel', 0.0),
        }
        
        # Compute mean values for property scores (filter out None values)
        if m0.get('QED'):
            metrics['qed_mean'] = np.mean(m0['QED']) if m0['QED'] else 0.0
        else:
            metrics['qed_mean'] = 0.0
            
        if m0.get('SA'):
            metrics['sa_mean'] = np.mean(m0['SA']) if m0['SA'] else 0.0
        else:
            metrics['sa_mean'] = 0.0
            
        if m0.get('NP'):
            metrics['np_mean'] = np.mean(m0['NP']) if m0['NP'] else 0.0
        else:
            metrics['np_mean'] = 0.0
            
        if m0.get('Solute'):
            metrics['logp_solute_mean'] = np.mean(m0['Solute']) if m0['Solute'] else 0.0
        else:
            metrics['logp_solute_mean'] = 0.0
            
        if m0.get('diverse'):
            metrics['diversity_mean'] = np.mean(m0['diverse']) if m0['diverse'] else 0.0
        else:
            metrics['diversity_mean'] = 0.0
            
        if m0.get('drugcand'):
            metrics['drugcand_mean'] = np.mean(m0['drugcand']) if m0['drugcand'] else 0.0
        else:
            metrics['drugcand_mean'] = 0.0
        
        return metrics
    except Exception as e:
        print(f"Warning: Error computing metrics: {e}")
        return {
            'valid': 0.0, 'unique': 0.0, 'novel': 0.0,
            'qed_mean': 0.0, 'sa_mean': 0.0, 'np_mean': 0.0,
            'logp_solute_mean': 0.0, 'diversity_mean': 0.0, 'drugcand_mean': 0.0
        }


def save_molecules_as_images(mols, img_dir, logp_values):
    """
    Save each valid molecule as individual PNG image with proper naming.
    
    Naming convention: mol_<index>_logp_<value>.png
    
    Args:
        mols: List of RDKit molecule objects
        img_dir: Directory to save images
        logp_values: List of LogP values (None for invalid molecules)
    
    Returns:
        saved_count: Number of images saved
    """
    saved_count = 0
    
    for idx, (mol, logp) in enumerate(zip(mols, logp_values)):
        if mol is not None and logp is not None:
            try:
                # Create filename with LogP value
                logp_str = f"{logp:.2f}".replace('.', 'p').replace('-', 'neg')
                filename = f"mol_{idx:04d}_logp_{logp_str}.png"
                filepath = os.path.join(img_dir, filename)
                
                # Save molecule image
                Draw.MolToFile(mol, filepath, size=(300, 300))
                saved_count += 1
            except Exception as e:
                print(f"Warning: Could not save molecule {idx}: {e}")
    
    return saved_count


def save_molecules_batch(mols, img_dir, class_name):
    """
    Save molecules using the same method as training (save_mol_img).
    Creates a single composite image of all valid molecules.
    """
    mol_f_name = os.path.join(img_dir, f'mol_batch_{class_name}.png')
    save_mol_img(mols, mol_f_name, is_test=True)
    return mol_f_name


def save_molecule_grid(mols, output_path, mols_per_row=5, max_mols=25):
    """Save molecule grid image with LogP labels"""
    valid_mols = [m for m in mols if m is not None][:max_mols]
    
    if not valid_mols:
        print('No valid molecules to visualize')
        return
    
    # Compute LogP for legends
    legends = []
    for m in valid_mols:
        try:
            logp = Crippen.MolLogP(m)
            legends.append(f'LogP: {logp:.2f}')
        except:
            legends.append('LogP: N/A')
    
    img = Draw.MolsToGridImage(
        valid_mols, 
        molsPerRow=mols_per_row,
        subImgSize=(300, 300),
        legends=legends
    )
    img.save(output_path)
    print(f'Saved molecule grid to {output_path}')


def main():
    args = parse_args()
    
    # Parse target class
    if args.class_type in ['hydrophilic', '0']:
        target_class = LOGP_CLASS_HYDROPHILIC
        class_name = 'hydrophilic'
        class_desc = 'hydrophilic (LogP < 0)'
    else:
        target_class = LOGP_CLASS_BALANCED
        class_name = 'balanced'
        class_desc = 'balanced (0 <= LogP <= 2)'
    
    # Setup output directories
    dirs = setup_output_directories(args, class_name)
    
    print('=' * 70)
    print('CONDITIONAL MOLECULE GENERATION')
    print('=' * 70)
    print(f'Target class:    {class_desc}')
    print(f'Num samples:     {args.num_samples}')
    print(f'Model dir:       {args.model_dir}')
    print(f'Output dir:      {dirs["run_dir"]}')
    print('=' * 70)
    
    # Load dataset (for encoders/decoders and novelty checking)
    print(f'\nLoading dataset from {args.mol_data_dir}...')
    data = SparseMolecularDataset()
    data.load(args.mol_data_dir, conditional=True)  # Need conditional=True to get smiles_set for novelty
    
    # Load generator
    G, model_epoch = load_generator(args, data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = G.to(device)
    
    print(f'\nGenerating {args.num_samples} {class_desc} molecules...')
    
    # Generate molecules (in batches with stochastic sampling)
    mols = generate_molecules(G, data, target_class, args.num_samples, args, device, batch_size=args.batch_size)
    
    # Compute statistics
    stats = compute_logp_stats(mols, target_class)
    
    # Compute all molecular metrics
    print('\nComputing molecular metrics (validity, uniqueness, novelty, QED, SA, NP, diversity, drug candidate)...')
    metrics = compute_all_metrics(mols, data)
    stats.update(metrics)  # Merge metrics into stats
    
    # Print results
    print('\n' + '=' * 70)
    print('GENERATION RESULTS')
    print('=' * 70)
    print(f'Total generated:   {stats["total"]}')
    print(f'Valid molecules:   {stats["valid"]} ({stats["validity_rate"]:.1f}%)')
    print(f'Class accuracy:    {stats["class_accuracy"]:.1f}% ({stats["correct_class"]}/{stats["valid"]})')
    print(f'')
    print(f'Molecular Quality Metrics:')
    print(f'  Validity:    {stats.get("valid", 0.0):.1f}%')
    print(f'  Uniqueness:  {stats.get("unique", 0.0):.1f}%')
    print(f'  Novelty:     {stats.get("novel", 0.0):.1f}%')
    print(f'')
    print(f'Drug-likeness Scores:')
    print(f'  QED:         {stats.get("qed_mean", 0.0):.3f}')
    print(f'  SA Score:    {stats.get("sa_mean", 0.0):.3f}')
    print(f'  NP Score:    {stats.get("np_mean", 0.0):.3f}')
    print(f'  Diversity:   {stats.get("diversity_mean", 0.0):.3f}')
    print(f'  Drug Cand:   {stats.get("drugcand_mean", 0.0):.3f}')
    print(f'')
    print(f'LogP Statistics (RAW values):')
    print(f'  Mean:  {stats["logp_mean"]:.3f}')
    print(f'  Std:   {stats["logp_std"]:.3f}')
    print(f'  Range: [{stats["logp_min"]:.3f}, {stats["logp_max"]:.3f}]')
    print(f'')
    print(f'LogP Statistics (NORMALIZED [0,1]):')
    print(f'  Mean:  {stats["logp_norm_mean"]:.3f}')
    print(f'  Std:   {stats["logp_norm_std"]:.3f}')
    
    # ==================== SAVE OUTPUTS ====================
    print('\n' + '-' * 70)
    print('SAVING OUTPUTS')
    print('-' * 70)
    
    # 1. Save individual molecule images
    print(f'\n[1/5] Saving individual molecule images to {dirs["img_dir"]}/')
    saved_count = save_molecules_as_images(mols, dirs['img_dir'], stats['logp_all'])
    print(f'      Saved {saved_count} molecule images')
    
    # 2. Save batch image (like training)
    print(f'\n[2/5] Saving batch image (training style) to {dirs["img_dir"]}/')
    batch_path = save_molecules_batch(mols, dirs['img_dir'], class_name)
    print(f'      Saved: {os.path.basename(batch_path)}')
    
    # 3. Save grid summary image
    print(f'\n[3/5] Saving grid summary to {dirs["grid_dir"]}/')
    grid_path = os.path.join(dirs['grid_dir'], f'{class_name}_grid_25.png')
    save_molecule_grid(mols, grid_path, mols_per_row=5, max_mols=25)
    
    # Also save larger grid
    grid_path_50 = os.path.join(dirs['grid_dir'], f'{class_name}_grid_50.png')
    save_molecule_grid(mols, grid_path_50, mols_per_row=10, max_mols=50)
    
    # 4. Save SMILES file
    print(f'\n[4/5] Saving SMILES to {dirs["smiles_dir"]}/')
    smiles_path = os.path.join(dirs['smiles_dir'], f'{class_name}_molecules.smi')
    with open(smiles_path, 'w') as f:
        f.write(f'# {class_desc} molecules\n')
        f.write(f'# Generated: {dirs["timestamp"]}\n')
        f.write(f'# Valid: {stats["valid"]}/{stats["total"]}, Class Accuracy: {stats["class_accuracy"]:.1f}%\n')
        f.write(f'# Validity: {stats.get("valid", 0.0):.1f}%, Uniqueness: {stats.get("unique", 0.0):.1f}%, Novelty: {stats.get("novel", 0.0):.1f}%\n')
        f.write(f'# QED: {stats.get("qed_mean", 0.0):.3f}, SA: {stats.get("sa_mean", 0.0):.3f}, NP: {stats.get("np_mean", 0.0):.3f}\n')
        f.write(f'# LogP: mean={stats["logp_mean"]:.2f}, std={stats["logp_std"]:.2f}\n')
        f.write(f'# Format: SMILES<tab>LogP\n\n')
        valid_count = 0
        for mol, logp in zip(mols, stats.get('logp_all', [])):
            if mol is not None and logp is not None:
                try:
                    smi = Chem.MolToSmiles(mol)
                    f.write(f'{smi}\t{logp:.4f}\n')
                    valid_count += 1
                except:
                    pass
    print(f'      Saved {valid_count} SMILES')
    
    # 5. Save generation config (with all metrics)
    print(f'\n[5/5] Saving generation config to {dirs["run_dir"]}/')
    config_path = save_generation_config(dirs, args, class_name, class_desc, model_epoch, stats)
    print(f'      Saved: config.json (includes all metrics)')
    
    # Final summary
    print('\n' + '=' * 70)
    print('OUTPUT SUMMARY')
    print('=' * 70)
    print(f'Run directory: {dirs["run_dir"]}')
    print(f'')
    print(f'├── img_dir/')
    print(f'│   ├── mol_XXXX_logp_X.XX.png  ({saved_count} individual images)')
    print(f'│   └── mol_batch_{class_name}.png')
    print(f'├── grid_dir/')
    print(f'│   ├── {class_name}_grid_25.png')
    print(f'│   └── {class_name}_grid_50.png')
    print(f'├── smiles/')
    print(f'│   └── {class_name}_molecules.smi')
    print(f'└── config.json')
    print('')
    print('Done!')


if __name__ == '__main__':
    main()

