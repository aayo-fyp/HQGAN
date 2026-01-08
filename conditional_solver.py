"""
Conditional Solver for LogP-based Molecular Generation.

Implements AC-GAN style training for conditional generation based on LogP classes:
- Class 0: Hydrophilic (LogP < 0)
- Class 1: Balanced (0 <= LogP <= 2)

Based on LGGAN paper: https://rlgm.github.io/papers/22.pdf
"""

from collections import defaultdict

import csv
import os
import time
import datetime
import numpy as np
import pandas as pd

import pennylane as qml
import random

import torch
import torch.nn.functional as F
import datetime
from utils.utils import *
from models.models import ConditionalGenerator, ConditionalDiscriminator
from data.sparse_molecular_dataset import SparseMolecularDataset, LOGP_CLASS_HYDROPHILIC, LOGP_CLASS_BALANCED
from utils.logger import Logger

from rdkit.Chem import Crippen

from frechetdist import frdist


def upper(m, a):
    """Build upper-triangle representation of adjacency one-hot tensor"""
    channels = m.shape[-1]
    res = torch.zeros((m.shape[0], 36, channels)).to(m.device).long()
    idx = torch.triu_indices(9, 9, offset=1)
    for i in range(m.shape[0]):
        for j in range(channels):
            tmp_m = m[i, :, :, j]
            res[i, :, j] = tmp_m[list(idx)]

    # Ensure last-dimension (channels) matches before concatenation
    a_channels = a.shape[-1]
    if res.shape[-1] != a_channels:
        target_ch = max(res.shape[-1], a_channels)
        if res.shape[-1] < target_ch:
            pad = torch.zeros((res.shape[0], res.shape[1], target_ch - res.shape[-1]), device=res.device, dtype=res.dtype)
            res = torch.cat((res, pad), dim=2)
        if a_channels < target_ch:
            pad = torch.zeros((a.shape[0], a.shape[1], target_ch - a_channels), device=a.device, dtype=a.dtype)
            a = torch.cat((a, pad), dim=2)

    return torch.cat((res, a), dim=1)


class ConditionalSolver(object):
    """
    Solver for training and testing Conditional Quantum-MolGAN.
    
    Implements AC-GAN style conditional generation for LogP-based classes.
    """

    def __init__(self, config, log=None):
        """Initialize configurations"""

        # Log
        self.log = log

        # Data loader - CONDITIONAL MODE
        self.data = SparseMolecularDataset()
        self.data.load(config.mol_data_dir, conditional=True)
        
        # Number of classes
        self.num_classes = self.data.num_classes
        self.class_embed_dim = getattr(config, 'class_embed_dim', 8)
        
        # Classification loss weight (AC-GAN)
        self.lambda_cls = getattr(config, 'lambda_cls', 1.0)

        # Quantum
        self.quantum = config.quantum
        self.layer = config.layer
        self.qubits = config.qubits
        self.gen_circuit = config.gen_circuit
        self.update_qc = config.update_qc
        self.qc_lr = config.qc_lr
        self.qc_pretrained = config.qc_pretrained

        # Model configurations
        self.z_dim = config.z_dim
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.la = config.lambda_wgan
        self.la_gp = config.lambda_gp
        self.post_method = config.post_method

        # Training configurations
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.num_steps = (len(self.data) // self.batch_size)
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.dropout = config.dropout
        self.gamma = config.gamma
        self.decay_every_epoch = config.decay_every_epoch

        # Critic
        if self.la > 0:
            self.n_critic = config.n_critic
        else:
            self.n_critic = 1
        self.critic_type = config.critic_type

        # Training or test
        self.mode = config.mode
        self.resume_epoch = config.resume_epoch

        # Testing configurations
        self.test_epoch = config.test_epoch
        self.test_sample_size = config.test_sample_size

        # Tensorboard
        self.use_tensorboard = config.use_tensorboard
        if self.mode == 'train' and config.use_tensorboard:
            self.logger = Logger(config.log_dir_path)

        # GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device: ', self.device, flush=True)

        # Directories
        self.log_dir_path = config.log_dir_path
        self.model_dir_path = config.model_dir_path
        self.img_dir_path = config.img_dir_path

        # CSV logging file path
        if config.mode == 'train':
            self.csv_log_path = os.path.join(config.log_dir_path, 'training_metrics.csv')
        else:
            self.csv_log_path = os.path.join(config.log_dir_path, 'test_metrics.csv')

        # Step size to save the model
        self.model_save_step = config.model_save_step

        # Build the model
        self.build_model()

        # Quantum circuit setup
        if config.quantum:
            if config.qc_pretrained:
                self.pretrained_qc_weights = pd.read_csv('results/quantum_circuit/molgan_red_weights.csv', header=None).iloc[-1, 1:].values
                self.gen_weights = torch.tensor(list(self.pretrained_qc_weights), requires_grad=True)
            else:
                self.gen_weights = torch.tensor(list(np.random.rand(config.layer*(config.qubits*2-1))*2*np.pi-np.pi), requires_grad=True)

            if self.update_qc:
                if self.qc_lr:
                    self.g_optimizer = torch.optim.RMSprop([
                        {'params': list(self.G.parameters())},
                        {'params': [self.gen_weights], 'lr': self.qc_lr}
                    ], lr=self.g_lr)
                else:
                    self.g_optimizer = torch.optim.RMSprop(list(self.G.parameters())+[self.gen_weights], self.g_lr)
            else:
                self.g_optimizer = torch.optim.RMSprop(list(self.G.parameters()), self.g_lr)

    def build_model(self):
        """Create conditional generator and discriminator"""

        # Conditional Generator
        self.G = ConditionalGenerator(
            self.g_conv_dim, 
            self.z_dim,
            self.data.vertexes,
            self.data.bond_num_types,
            self.data.atom_num_types,
            self.dropout,
            num_classes=self.num_classes,
            class_embed_dim=self.class_embed_dim
        )

        # Conditional Discriminator with Auxiliary Classifier
        self.D = ConditionalDiscriminator(
            self.d_conv_dim, 
            self.m_dim, 
            self.b_dim - 1,
            num_classes=self.num_classes,
            dropout_rate=self.dropout
        )

        # Optimizers
        self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), self.g_lr)
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), self.d_lr)

        # Print networks
        self.print_network(self.G, 'Conditional Generator', self.log)
        self.print_network(self.D, 'Conditional Discriminator', self.log)

        # Move to GPU
        self.G.to(self.device)
        self.D.to(self.device)

    @staticmethod
    def print_network(model, name, log=None):
        """Print out the network information"""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        if log is not None:
            log.info(model)
            log.info(name)
            log.info("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator"""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_dir_path, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_dir_path, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def load_gen_weights(self, resume_iters):
        """Restore the trained quantum circuit"""
        weights_pth = os.path.join(self.model_dir_path, 'molgan_red_weights.csv')
        weights = pd.read_csv(weights_pth, header=None).iloc[resume_iters-1, 1:].values
        self.gen_weights = torch.tensor(list(weights), requires_grad=True)

    def update_lr(self, gamma):
        """Decay learning rates"""
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] *= gamma
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] *= gamma

    def reset_grad(self):
        """Reset the gradient buffers"""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2"""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y, inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors"""
        out = torch.zeros(list(labels.size()) + [dim]).to(self.device)
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
        return out

    def sample_z(self, batch_size):
        """Sample the random noise"""
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    @staticmethod
    def postprocess(inputs, method, temperature=1.0):
        """Convert the probability matrices into label matrices"""
        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]
        
        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))/temperature, hard=False).view(e_logits.size()) for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))/temperature, hard=True).view(e_logits.size()) for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits/temperature, -1) for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]

    def train_and_validate(self):
        """Train and validate function"""
        self.start_time = time.time()

        # Start training from scratch or resume training
        start_epoch = 0
        if self.resume_epoch is not None and self.mode == 'train':
            start_epoch = self.resume_epoch
            self.restore_model(self.resume_epoch)
            if self.quantum:
                self.load_gen_weights(self.resume_epoch)
        elif self.test_epoch is not None and self.mode == 'test':
            self.restore_model(self.test_epoch)
            if self.quantum:
                self.load_gen_weights(self.test_epoch)
        else:
            print('Training From Scratch...')

        # Start training loop or test phase
        if self.mode == 'train':
            print('Start conditional training...')
            print('Classes: 0=Hydrophilic (LogP<0), 1=Balanced (0<=LogP<=2)')
            for i in range(start_epoch, self.num_epochs):
                self.train_or_valid(epoch_i=i, train_val_test='train')
                self.train_or_valid(epoch_i=i, train_val_test='val')
        elif self.mode == 'test':
            print('Start testing...')
            assert (self.resume_epoch is not None or self.test_epoch is not None)
            self.train_or_valid(epoch_i=start_epoch, train_val_test='val')
        else:
            raise NotImplementedError

    def get_gen_mols(self, n_hat, e_hat, method):
        """Convert edges and nodes matrices into molecules"""
        (edges_hard, nodes_hard) = self.postprocess((e_hat, n_hat), method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True) for e_, n_ in zip(edges_hard, nodes_hard)]
        return mols

    def compute_class_accuracy(self, mols, target_labels):
        """
        Compute accuracy of generated molecules matching target LogP class.
        
        Returns:
            accuracy: Fraction of valid molecules with correct LogP class
            per_class_accuracy: Dict with accuracy per class
            logp_stats: Dict with raw and normalized LogP statistics
        """
        correct = 0
        total_valid = 0
        class_correct = {0: 0, 1: 0}
        class_total = {0: 0, 1: 0}
        logp_raw_values = []
        
        target_labels_np = target_labels.cpu().numpy() if torch.is_tensor(target_labels) else target_labels
        
        for mol, target in zip(mols, target_labels_np):
            if mol is not None:
                try:
                    logp = Crippen.MolLogP(mol)
                    logp_raw_values.append(logp)
                    total_valid += 1
                    class_total[target] += 1
                    
                    # Check if LogP matches target class (using RAW LogP thresholds)
                    if target == 0 and logp < 0:  # Hydrophilic
                        correct += 1
                        class_correct[target] += 1
                    elif target == 1 and 0 <= logp <= 2:  # Balanced
                        correct += 1
                        class_correct[target] += 1
                except:
                    pass
        
        accuracy = correct / total_valid if total_valid > 0 else 0
        per_class_acc = {
            'hydrophilic': class_correct[0] / class_total[0] if class_total[0] > 0 else 0,
            'balanced': class_correct[1] / class_total[1] if class_total[1] > 0 else 0
        }
        
        # Compute LogP statistics (both raw and normalized)
        logp_stats = {}
        if logp_raw_values:
            logp_raw = np.array(logp_raw_values)
            # Normalized LogP: maps from [-2.12, 6.04] to [0, 1] (same as utils.py)
            logp_norm = np.clip((logp_raw - (-2.12178879609)) / (6.0429063424 - (-2.12178879609)), 0.0, 1.0)
            
            logp_stats = {
                'logp_raw_mean': np.mean(logp_raw),
                'logp_raw_std': np.std(logp_raw),
                'logp_norm_mean': np.mean(logp_norm),
                'logp_norm_std': np.std(logp_norm),
            }
        else:
            logp_stats = {
                'logp_raw_mean': 0,
                'logp_raw_std': 0,
                'logp_norm_mean': 0,
                'logp_norm_std': 0,
            }
        
        return accuracy, per_class_acc, logp_stats

    def log_metrics_to_csv(self, epoch_i, step, losses, scores, et=0):
        """Log all training metrics to CSV file"""
        file_exists = os.path.exists(self.csv_log_path)
        
        with open(self.csv_log_path, 'a', newline='') as file:
            writer = csv.writer(file)
            
            if not file_exists:
                headers = ['epoch', 'step', 'elapsed_time']
                loss_headers = ['D/loss_real', 'D/loss_fake', 'D/loss_gp', 'D/loss', 
                               'D/loss_cls', 'G/loss', 'G/loss_cls', 'FD/bond', 'FD/bond_atom']
                headers.extend(loss_headers)
                # Score headers including both raw and normalized LogP
                score_headers = ['valid', 'unique', 'novel', 'NP', 'QED', 'Solute', 'SA', 
                                'diverse', 'drugcand', 'class_acc', 'hydrophilic_acc', 'balanced_acc',
                                'logp_raw_mean', 'logp_raw_std', 'logp_norm_mean', 'logp_norm_std']
                headers.extend(score_headers)
                writer.writerow(headers)
            
            row = [epoch_i + 1, step, et]
            
            loss_keys = ['D/loss_real', 'D/loss_fake', 'D/loss_gp', 'D/loss', 
                        'D/loss_cls', 'G/loss', 'G/loss_cls', 'FD/bond', 'FD/bond_atom']
            for key in loss_keys:
                if key in losses and len(losses[key]) > 0:
                    row.append(np.mean(losses[key]))
                else:
                    row.append('')
            
            score_keys = ['valid', 'unique', 'novel', 'NP', 'QED', 'Solute', 'SA', 
                         'diverse', 'drugcand', 'class_acc', 'hydrophilic_acc', 'balanced_acc',
                         'logp_raw_mean', 'logp_raw_std', 'logp_norm_mean', 'logp_norm_std']
            for key in score_keys:
                if key in scores and len(scores[key]) > 0:
                    row.append(np.mean(scores[key]))
                else:
                    row.append('')
            
            writer.writerow(row)

    def save_checkpoints(self, epoch_i):
        """Store the models and quantum circuit"""
        G_path = os.path.join(self.model_dir_path, '{}-G.ckpt'.format(epoch_i + 1))
        D_path = os.path.join(self.model_dir_path, '{}-D.ckpt'.format(epoch_i + 1))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        
        # Save quantum weights
        if self.quantum:
            with open(os.path.join(self.model_dir_path, 'molgan_red_weights.csv'), 'a') as file:
                writer = csv.writer(file)
                writer.writerow([str(epoch_i)]+list(self.gen_weights.detach().numpy()))
        
        print('Saved model checkpoints into {}...'.format(self.model_dir_path))
        if self.log is not None:
            self.log.info('Saved model checkpoints into {}...'.format(self.model_dir_path))

    def train_or_valid(self, epoch_i, train_val_test='val'):
        """Train or validate function with conditional generation"""
        
        cur_la = self.la

        # Recordings
        losses = defaultdict(list)
        scores = defaultdict(list)

        # Iterations
        the_step = self.num_steps
        if train_val_test == 'val':
            if self.mode == 'train':
                the_step = 1
                print('[Validating]')
            elif self.mode == 'test':
                the_step = 1
                print('[Testing]')
            else:
                raise NotImplementedError

        for a_step in range(the_step):

            ########## Get batch data WITH CLASS LABELS ##########
            if train_val_test == 'val' and not self.quantum:
                if self.test_sample_size is None:
                    batch = self.data.next_validation_batch()
                    batch_size = batch[3].shape[0]
                else:
                    batch = self.data.next_validation_batch(self.test_sample_size)
                    batch_size = self.test_sample_size
                z = self.sample_z(batch_size)
            elif train_val_test == 'train' and not self.quantum:
                batch = self.data.next_train_batch(self.batch_size)
                batch_size = self.batch_size
                z = self.sample_z(self.batch_size)
            elif train_val_test == 'val' and self.quantum:
                if self.test_sample_size is None:
                    batch = self.data.next_validation_batch()
                    batch_size = batch[3].shape[0]
                else:
                    batch = self.data.next_validation_batch(self.test_sample_size)
                    batch_size = self.test_sample_size
                sample_list = [torch.stack(self.gen_circuit(self.gen_weights)) for i in range(batch_size)]
            elif train_val_test == 'train' and self.quantum:
                batch = self.data.next_train_batch(self.batch_size)
                batch_size = self.batch_size
                sample_list = [torch.stack(self.gen_circuit(self.gen_weights)) for i in range(self.batch_size)]
            else:
                raise NotImplementedError

            # Unpack batch (conditional mode includes class labels)
            mols, _, _, a, x, _, _, _, _, real_class_labels = batch

            ########## Preprocess input data ##########
            a = torch.from_numpy(a).to(self.device).long()
            x = torch.from_numpy(x).to(self.device).long()
            a_tensor = self.label2onehot(a, self.b_dim)
            x_tensor = self.label2onehot(x, self.m_dim)
            
            # Real class labels
            real_class_labels = torch.from_numpy(real_class_labels).to(self.device).long()
            
            # Sample target class labels for generator
            target_class_labels = torch.from_numpy(
                self.data.sample_class_labels(batch_size)
            ).to(self.device).long()

            if self.quantum:
                z = torch.stack(tuple(sample_list)).to(self.device).float()
            else:
                z = torch.from_numpy(z).to(self.device).float()

            # Tensorboard
            loss_tb = {}

            # Current step
            cur_step = self.num_steps * epoch_i + a_step

            ########## Train the discriminator ##########
            
            # Forward pass on real data
            logits_real, cls_logits_real, features_real = self.D(a_tensor, None, x_tensor)
            
            # Generate fake molecules conditioned on target class
            edges_logits, nodes_logits = self.G(z, target_class_labels)
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
            
            # Forward pass on fake data
            logits_fake, cls_logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)

            # Gradient penalty
            eps = torch.rand(logits_real.size(0), 1, 1, 1).to(self.device)
            x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
            x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
            grad0, _, _ = self.D(x_int0, None, x_int1)
            grad_penalty = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(grad0, x_int1)

            # WGAN losses
            d_loss_real = torch.mean(logits_real)
            d_loss_fake = torch.mean(logits_fake)
            
            # Classification loss on REAL samples (AC-GAN)
            d_loss_cls = F.cross_entropy(cls_logits_real, real_class_labels)
            
            # Total discriminator loss
            loss_D = -d_loss_real + d_loss_fake + self.la_gp * grad_penalty + self.lambda_cls * d_loss_cls

            # Record losses
            losses['D/loss_real'].append(d_loss_real.item())
            losses['D/loss_fake'].append(d_loss_fake.item())
            losses['D/loss_gp'].append(grad_penalty.item())
            losses['D/loss_cls'].append(d_loss_cls.item())
            losses['D/loss'].append(loss_D.item())

            loss_tb['D/loss_real'] = d_loss_real.item()
            loss_tb['D/loss_fake'] = d_loss_fake.item()
            loss_tb['D/loss_gp'] = grad_penalty.item()
            loss_tb['D/loss_cls'] = d_loss_cls.item()
            loss_tb['D/loss'] = loss_D.item()

            # Optimize discriminator
            if train_val_test == 'train':
                if self.critic_type == 'D':
                    if (cur_step == 0) or (cur_step % self.n_critic != 0):
                        self.reset_grad()
                        loss_D.backward()
                        self.d_optimizer.step()
                else:
                    if (cur_step != 0) and (cur_step % self.n_critic == 0):
                        self.reset_grad()
                        loss_D.backward()
                        self.d_optimizer.step()

            ########## Train the generator ##########

            # Generate fake molecules
            edges_logits, nodes_logits = self.G(z, target_class_labels)
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
            
            logits_fake, cls_logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)

            # WGAN generator loss (fool discriminator)
            loss_G_adv = -torch.mean(logits_fake)
            
            # Classification loss on FAKE samples (encourage generator to produce correct class)
            loss_G_cls = F.cross_entropy(cls_logits_fake, target_class_labels)
            
            # Total generator loss
            loss_G = loss_G_adv + self.lambda_cls * loss_G_cls

            losses['G/loss'].append(loss_G_adv.item())
            losses['G/loss_cls'].append(loss_G_cls.item())

            loss_tb['G/loss'] = loss_G_adv.item()
            loss_tb['G/loss_cls'] = loss_G_cls.item()

            print('d_loss {:.2f} d_fake {:.2f} d_real {:.2f} d_cls {:.2f} g_loss: {:.2f} g_cls: {:.2f}'.format(
                loss_D.item(), d_loss_fake.item(), d_loss_real.item(), d_loss_cls.item(), 
                loss_G_adv.item(), loss_G_cls.item()))
            print('======================= {} =============================='.format(datetime.datetime.now()), flush=True)

            # Optimize generator
            if train_val_test == 'train':
                if self.critic_type == 'D':
                    if (cur_step != 0) and (cur_step % self.n_critic) == 0:
                        self.reset_grad()
                        loss_G.backward()
                        self.g_optimizer.step()
                else:
                    if (cur_step == 0) or (cur_step % self.n_critic != 0):
                        self.reset_grad()
                        loss_G.backward()
                        self.g_optimizer.step()

            if train_val_test == 'train' and self.use_tensorboard:
                for tag, value in loss_tb.items():
                    self.logger.scalar_summary(tag, value, cur_step)

            ########## Frechet distribution ##########
            (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
            edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
            R = [list(a[i].reshape(-1).to('cpu')) for i in range(batch_size)]
            F_dist = [list(edges_hard[i].reshape(-1).to('cpu')) for i in range(batch_size)]
            fd_bond = frdist(R, F_dist)

            R = [list(x[i].to('cpu')) + list(a[i].reshape(-1).to('cpu')) for i in range(batch_size)]
            F_dist = [list(nodes_hard[i].to('cpu')) + list(edges_hard[i].reshape(-1).to('cpu')) for i in range(batch_size)]
            fd_bond_atom = frdist(R, F_dist)

            loss_tb['FD/bond'] = fd_bond
            loss_tb['FD/bond_atom'] = fd_bond_atom
            losses['FD/bond'].append(fd_bond)
            losses['FD/bond_atom'].append(fd_bond_atom)

            if train_val_test == 'train' and self.use_tensorboard:
                for tag, value in loss_tb.items():
                    self.logger.scalar_summary(tag, value, cur_step)

            ########## Miscellaneous ##########

            # Decay learning rates
            if epoch_i != 0 and self.decay_every_epoch:
                if a_step == 0 and (epoch_i+1) % self.decay_every_epoch == 0:
                    self.update_lr(self.gamma)

            # Get scores
            if a_step % 10 == 0:
                gen_mols = self.get_gen_mols(nodes_logits, edges_logits, self.post_method)
                m0, m1 = all_scores(gen_mols, self.data, norm=True)
                
                for k, v in m1.items():
                    scores[k].append(v)
                for k, v in m0.items():
                    # Safer computation: handle empty arrays gracefully (from original QuantumMolGAN)
                    v_array = np.array(v)
                    nonzero_v = v_array[np.nonzero(v_array)] if len(v_array) > 0 else np.array([])
                    scores[k].append(nonzero_v.mean() if len(nonzero_v) > 0 else np.nan)

                # Compute class accuracy and LogP statistics
                class_acc, per_class_acc, logp_stats = self.compute_class_accuracy(gen_mols, target_class_labels)
                scores['class_acc'].append(class_acc * 100)
                scores['hydrophilic_acc'].append(per_class_acc['hydrophilic'] * 100)
                scores['balanced_acc'].append(per_class_acc['balanced'] * 100)
                
                # LogP statistics (raw and normalized)
                scores['logp_raw_mean'].append(logp_stats['logp_raw_mean'])
                scores['logp_raw_std'].append(logp_stats['logp_raw_std'])
                scores['logp_norm_mean'].append(logp_stats['logp_norm_mean'])
                scores['logp_norm_std'].append(logp_stats['logp_norm_std'])

                # Save checkpoints
                if self.mode == 'train':
                    if (epoch_i + 1) % self.model_save_step == 0:
                        self.save_checkpoints(epoch_i=epoch_i)

                # Save molecule images
                mol_f_name = os.path.join(self.img_dir_path, 'mol-{}.png'.format(epoch_i))
                save_mol_img(gen_mols, mol_f_name, is_test=self.mode == 'test')

                # Calculate elapsed time
                et_seconds = time.time() - self.start_time

                # Log metrics to CSV
                self.log_metrics_to_csv(epoch_i, a_step, losses, scores, et=et_seconds)

                # Print out training information
                et = str(datetime.timedelta(seconds=et_seconds))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]:".format(et, epoch_i + 1, self.num_epochs)

                is_first = True
                for tag, value in losses.items():
                    if is_first:
                        log += "\n{}: {:.2f}".format(tag, np.mean(value))
                        is_first = False
                    else:
                        log += ", {}: {:.2f}".format(tag, np.mean(value))
                        
                is_first = True
                for tag, value in scores.items():
                    if is_first:
                        log += "\n{}: {:.2f}".format(tag, np.mean(value))
                        is_first = False
                    else:
                        log += ", {}: {:.2f}".format(tag, np.mean(value))
                print(log)

                if self.log is not None:
                    self.log.info(log)

    def generate_conditional(self, target_class, num_samples=100):
        """
        Generate molecules of a specific class.
        
        Args:
            target_class: 0 for hydrophilic (LogP < 0), 1 for balanced (0 <= LogP <= 2)
            num_samples: Number of molecules to generate
            
        Returns:
            mols: List of RDKit molecule objects
            logp_values: List of LogP values for generated molecules
        """
        self.G.eval()
        
        with torch.no_grad():
            # Create target labels
            target_labels = torch.full((num_samples,), target_class, dtype=torch.long).to(self.device)
            
            # Generate noise
            if self.quantum:
                z = torch.stack([torch.stack(self.gen_circuit(self.gen_weights)) for _ in range(num_samples)]).to(self.device).float()
            else:
                z = torch.from_numpy(self.sample_z(num_samples)).to(self.device).float()
            
            # Generate molecules
            edges_logits, nodes_logits = self.G(z, target_labels)
            
            # Convert to molecules
            mols = self.get_gen_mols(nodes_logits, edges_logits, 'hard_gumbel')
            
            # Compute LogP for valid molecules
            logp_values = []
            for mol in mols:
                if mol is not None:
                    try:
                        logp = Crippen.MolLogP(mol)
                        logp_values.append(logp)
                    except:
                        logp_values.append(None)
                else:
                    logp_values.append(None)
        
        self.G.train()
        
        return mols, logp_values

