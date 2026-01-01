# Implementation Roadmap: HQ-Cycle-MolGAN with User-Guided Conditioning

This document outlines the implementation plan to extend the base HQGAN implementation to match the **Hybrid Quantum Cycle MolGAN (HQ-Cycle-MolGAN)** architecture from [pp2.pdf](pp2.pdf), and introduces our novel **User-Guided Conditioning Component** for property-targeted drug discovery.

## Missing Components from Base Paper

### 1. Cycle Component Architecture

**Current State**: Base implementation has Generator (G) and Discriminator (D) only.

**Missing**: Cycle-consistency architecture with bidirectional generators.

**Required Components**:
- **Generator G_AB**: Generates molecules from domain A to domain B
- **Generator G_BA**: Generates molecules from domain B to domain A (cycle generator)
- **Cycle Loss**: Enforces `G_BA(G_AB(A)) ≈ A` and `G_AB(G_BA(B)) ≈ B`
- **Identity Loss** (optional): `G_BA(A) ≈ A` when domains are similar

**Implementation Steps**:
1. Create `CycleGenerator` class in `models/models.py` that wraps two generators
2. Add cycle loss computation in `solver.py`:
   ```python
   # Forward cycle: A -> B -> A
   fake_B = G_AB(real_A)
   rec_A = G_BA(fake_B)
   cycle_loss_A = ||rec_A - real_A||_1
   
   # Backward cycle: B -> A -> B  
   fake_A = G_BA(real_B)
   rec_B = G_AB(fake_A)
   cycle_loss_B = ||rec_B - real_B||_1
   
   cycle_loss = lambda_cycle * (cycle_loss_A + cycle_loss_B)
   ```
3. Update generator loss: `L_G = L_GAN + cycle_loss + identity_loss`
4. Both G_AB and G_BA should use quantum circuits for noise generation when `quantum=True`

### 2. Enhanced Quantum Circuit Architecture

**Current State**: Basic variational quantum circuit with RY, RZ, and CNOT gates.

**Missing**: Paper mentions more sophisticated quantum architectures for HQNN components.

**Required Enhancements**:
- Strongly Entangling Layers (SEL) template for better expressivity
- Multiple quantum layers with residual connections
- Quantum feature maps for encoding classical inputs

**Implementation Steps**:
1. Replace basic circuit in `main.py` with PennyLane's `StronglyEntanglingLayers`
2. Add quantum feature embedding for conditional inputs
3. Support multiple quantum circuit configurations (layer counts, entanglement patterns)

### 3. Domain Separation Strategy

**Current State**: Single domain (all molecules from same dataset).

**Missing**: Paper discusses domain A and domain B for cycle training.

**Required Strategy**:
- Split dataset into two domains based on:
  - Property thresholds (e.g., QED > 0.5 vs QED ≤ 0.5)
  - Molecular size (large vs small)
  - Functional groups or scaffolds
- Create `SparseMolecularDatasetPair` class for paired/unpaired domain data

**Implementation Steps**:
1. Extend `data/sparse_molecular_dataset.py` with domain splitting methods
2. Add `next_domain_batch()` method returning (A, B) pairs
3. Implement property-based domain assignment

### 4. Training Stability Improvements

**Current State**: Basic WGAN with gradient penalty.

**Missing**: Cycle consistency stabilizes training as noted in paper.

**Required Components**:
- Learning rate scheduling specific to cycle training
- Separate optimizers for G_AB, G_BA, D_A, D_B
- Gradient accumulation for quantum circuits (if needed)

## User-Guided Conditioning Component (Novel Contribution)

### Overview

A conditioning mechanism that allows users to specify target molecular properties (e.g., QED range, LogP, SA score) to guide generation toward desired property profiles.

### Architecture Design

#### 1. Conditional Generator

**Component**: `ConditionalGenerator` extending base Generator

**Inputs**:
- Noise vector `z`: `(batch_size, z_dim)`
- Property condition `c`: `(batch_size, n_properties)` where properties include:
  - QED target (0-1)
  - LogP target (float)
  - SA score target (1-10, lower is better)
  - Molecular weight range (min, max)
  - Optional: scaffold or substructure constraints

**Architecture**:
```
z: (B, z_dim) ─┐
               ├─→ Concatenate → (B, z_dim + n_properties)
c: (B, n_props)┘                ↓
                         Conditional Dense Layers
                         ↓
                    Generator Head (edges + nodes)
```

**Implementation**:
```python
class ConditionalGenerator(Generator):
    def __init__(self, ..., n_conditions=5):
        super().__init__(...)
        self.condition_embedding = nn.Linear(n_conditions, condition_dim)
        self.conditional_dense = MultiDenseLayers(
            z_dim + condition_dim, conv_dims, activation, dropout
        )
    
    def forward(self, z, conditions):
        c_emb = self.condition_embedding(conditions)
        z_cond = torch.cat([z, c_emb], dim=1)
        return super().forward_from_embedding(z_cond)
```

#### 2. Property Predictor Network

**Component**: Auxiliary network that predicts properties from generated molecules

**Purpose**: Provides gradient signal for property targeting during training

**Architecture**:
- Input: Generated molecular graph (edges + nodes)
- Output: Predicted properties `(batch_size, n_properties)`
- Loss: MSE between predicted and target properties

**Implementation**:
```python
class PropertyPredictor(nn.Module):
    def __init__(self, m_dim, b_dim, n_properties):
        # Reuse discriminator backbone
        self.gcn = GraphConvolution(...)
        self.agg = GraphAggregation(...)
        self.property_head = nn.Linear(hidden_dim, n_properties)
    
    def forward(self, edges, nodes):
        features = self.gcn(nodes, edges)
        graph_features = self.agg(nodes, features)
        return self.property_head(graph_features)
```

#### 3. Conditional Quantum Circuit

**Component**: Quantum circuit that encodes conditions into quantum state

**Approach**: Use amplitude/angle embedding for conditions in quantum circuit

**Implementation**:
```python
@qml.qnode(dev, interface='torch', diff_method='backprop')
def conditional_gen_circuit(w, conditions):
    # Encode conditions into quantum state
    qml.AngleEmbedding(conditions, wires=range(n_qubits))
    
    # Variational layers
    for l in range(config.layer):
        qml.StronglyEntanglingLayers(w[l], wires=range(n_qubits))
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
```

#### 4. Conditioning Loss Function

**Component**: Combined loss ensuring generated molecules match target properties

**Loss Components**:
```python
# Property matching loss
pred_properties = property_predictor(fake_edges, fake_nodes)
property_loss = MSE(pred_properties, target_conditions)

# Cycle consistency with conditions
fake_B_cond = G_AB(real_A, conditions_B)
rec_A_cond = G_BA(fake_B_cond, conditions_A)
cycle_loss = ||rec_A_cond - real_A|| + property_loss

# Combined generator loss
L_G = L_GAN + lambda_cycle * cycle_loss + lambda_prop * property_loss
```

### User Interface Design

#### Configuration Format

```python
# Example user condition specification
user_conditions = {
    'qed': {'target': 0.7, 'tolerance': 0.1},  # QED between 0.6-0.8
    'logp': {'target': 2.5, 'tolerance': 1.0},  # LogP between 1.5-3.5
    'sa_score': {'target': 3.0, 'tolerance': 2.0},  # SA between 1-5
    'mol_weight': {'min': 200, 'max': 500},  # MW between 200-500 Da
    'scaffold': 'benzene_ring'  # Optional substructure constraint
}
```

#### Training Mode

**Option 1: Pre-training + Fine-tuning**
1. Pre-train base HQ-Cycle-MolGAN without conditioning
2. Add conditioning layers and fine-tune on property-targeted data

**Option 2: Joint Training**
1. Train conditioning components alongside base model
2. Use curriculum learning: start with weak conditioning, increase weight over epochs

#### Inference Mode

```python
# Generate molecules with specific property targets
generated_mols = generate_conditional(
    model_path='checkpoints/hq_cycle_molgan.ckpt',
    conditions=user_conditions,
    num_samples=100,
    temperature=0.8  # Controls exploration vs exploitation
)
```

## Implementation Priority

### Phase 1: Base Cycle Architecture
1. Implement cycle generators (G_AB, G_BA)
2. Add cycle loss to training loop
3. Test on property-based domain splits
4. Validate cycle consistency metrics

### Phase 2: Enhanced Quantum Components  
1. Upgrade quantum circuits with SEL templates
2. Implement quantum feature embedding
3. Test quantum circuit expressivity improvements

### Phase 3: User-Guided Conditioning (Novel)
1. Implement ConditionalGenerator
2. Add PropertyPredictor network
3. Integrate conditional quantum circuits
4. Design and implement conditioning loss
5. Create user interface for condition specification

### Phase 4: Integration & Validation
1. Combine cycle + conditioning components
2. Hyperparameter tuning for lambda weights
3. Benchmark against base paper metrics (QED, SA, LogP)
4. Ablation studies on conditioning effectiveness

## Key Files to Modify/Create

### New Files
- `models/cycle_models.py` - Cycle generator architectures
- `models/conditional_generator.py` - Conditional generator
- `models/property_predictor.py` - Property prediction network
- `utils/conditioning.py` - Conditioning utilities and loss functions
- `data/domain_dataset.py` - Domain splitting and pairing logic

### Modified Files
- `solver.py` - Add cycle training loop and conditioning losses
- `main.py` - Add cycle and conditioning configuration options
- `models/models.py` - Extend base generator with conditional support
- `data/sparse_molecular_dataset.py` - Add domain splitting methods

## Expected Improvements

Based on the base paper's results (30% QED improvement with cycle architecture), adding user-guided conditioning should provide:

- **Property Targeting**: Generate molecules within specified property ranges
- **Discovery Efficiency**: Reduce search space by 50-70% for targeted properties
- **Quality Metrics**: Maintain or improve QED, SA, LogP scores while meeting constraints
- **Training Stability**: Cycle consistency + conditioning provides stronger gradients

## References

- Base Paper: "Hybrid Quantum Cycle Generative Adversarial Network for Small Molecule Generation" (pp2.pdf)
- CycleGAN: [Zhu et al., 2017](https://arxiv.org/abs/1703.10593)
- MolGAN: [De Cao & Kipf, 2018](https://arxiv.org/abs/1805.11973)
- Quantum GAN: [Kao et al., 2023](https://arxiv.org/abs/2210.16823)

