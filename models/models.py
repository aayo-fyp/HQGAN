import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers import GraphConvolution, GraphAggregation, MultiGraphConvolutionLayers, MultiDenseLayers

# decoder_adj in MolGAN/models/__init__.py
# Implementation-MolGAN-PyTorch/models_gan.py Generator
class Generator(nn.Module):
    """Generator network of MolGAN"""

    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout_rate):
        super(Generator, self).__init__()
        self.conv_dims = conv_dims
        self.z_dim = z_dim
        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        self.dropout_rate = dropout_rate

        self.activation_f = nn.Tanh()
        self.multi_dense_layers = MultiDenseLayers(z_dim, conv_dims, self.activation_f)
        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        output = self.multi_dense_layers(x)
        edges_logits = self.edges_layer(output).view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits


# encoder_rgcn in MolGAN/model/__init__.py
# MolGAN/models/gan.py GraphGANModel.D_x
# Implementation-MolGAN-PyTorch/models_gan.py Discriminator
class Discriminator(nn.Module):
    """Discriminator network of MolGAN"""

    def __init__(self, conv_dims, m_dim, b_dim, with_features=False, f_dim=0, dropout_rate=0.):
        super(Discriminator, self).__init__()
        self.conv_dims = conv_dims
        self.m_dim = m_dim
        self.b_dim = b_dim
        self.with_features = with_features
        self.f_dim = f_dim
        self.dropout_rate = dropout_rate

        self.activation_f = nn.Tanh()
        # line #6
        graph_conv_dim, aux_dim, linear_dim = conv_dims
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, with_features, f_dim, dropout_rate)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1] + m_dim, aux_dim, self.activation_f, with_features, f_dim, dropout_rate)
        self.multi_dense_layers = MultiDenseLayers(aux_dim, linear_dim, self.activation_f, dropout_rate)
        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adjacency_tensor, hidden, node, activation=None):
        adj = adjacency_tensor[:, :, :, 1:].permute(0, 3, 1, 2)
        h = self.gcn_layer(node, adj, hidden)
        h = self.agg_layer(node, h, hidden)
        #h = self.agg_layer(h, node, hidden)
        h = self.multi_dense_layers(h)

        output = self.output_layer(h)
        output = activation(output) if activation is not None else output

        return output, h


# ============================================================================
# CONDITIONAL MODELS FOR LOGP-BASED GENERATION (AC-GAN STYLE)
# ============================================================================

class ConditionalGenerator(nn.Module):
    """
    Conditional Generator for LogP-based molecular generation.
    
    Takes noise z and class label c as input, generates molecules
    conditioned on the specified LogP class (hydrophilic or balanced).
    
    Based on AC-GAN framework from LGGAN paper.
    """

    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout_rate,
                 num_classes=2, class_embed_dim=8):
        """
        Args:
            conv_dims: List of dense layer dimensions
            z_dim: Dimension of noise vector
            vertexes: Maximum number of atoms in molecule
            edges: Number of edge/bond types
            nodes: Number of node/atom types
            dropout_rate: Dropout probability
            num_classes: Number of LogP classes (default: 2 for hydrophilic/balanced)
            class_embed_dim: Dimension of class embedding
        """
        super(ConditionalGenerator, self).__init__()
        self.conv_dims = conv_dims
        self.z_dim = z_dim
        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim

        # Class embedding layer
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        
        # Activation
        self.activation_f = nn.Tanh()
        
        # Dense layers: input is z concatenated with class embedding
        input_dim = z_dim + class_embed_dim
        self.multi_dense_layers = MultiDenseLayers(input_dim, conv_dims, self.activation_f)
        
        # Output layers for molecular graph
        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, z, class_labels):
        """
        Generate molecular graph conditioned on class label.
        
        Args:
            z: Noise vector of shape (batch_size, z_dim)
            class_labels: Class labels of shape (batch_size,) with values 0 or 1
                         0 = hydrophilic (LogP < 0)
                         1 = balanced (0 <= LogP <= 2)
        
        Returns:
            edges_logits: Edge logits of shape (batch_size, vertexes, vertexes, edges)
            nodes_logits: Node logits of shape (batch_size, vertexes, nodes)
        """
        # Embed class labels
        c_emb = self.class_embedding(class_labels)  # (batch_size, class_embed_dim)
        
        # Concatenate noise and class embedding
        z_cond = torch.cat([z, c_emb], dim=1)  # (batch_size, z_dim + class_embed_dim)
        
        # Generate molecular graph
        output = self.multi_dense_layers(z_cond)
        
        # Edge logits
        edges_logits = self.edges_layer(output).view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2  # Symmetric
        edges_logits = self.dropout(edges_logits.permute(0, 2, 3, 1))

        # Node logits
        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator with Auxiliary Classifier (AC-GAN style).
    
    Outputs:
    1. Real/Fake score (WGAN)
    2. Class prediction (auxiliary classifier for LogP class)
    
    The auxiliary classifier helps the generator learn class-specific features
    by providing additional gradient signal.
    """

    def __init__(self, conv_dims, m_dim, b_dim, num_classes=2,
                 with_features=False, f_dim=0, dropout_rate=0.):
        """
        Args:
            conv_dims: [graph_conv_dim, aux_dim, linear_dim]
            m_dim: Number of atom types
            b_dim: Number of bond types
            num_classes: Number of LogP classes (default: 2)
            with_features: Whether to use additional features
            f_dim: Feature dimension
            dropout_rate: Dropout probability
        """
        super(ConditionalDiscriminator, self).__init__()
        self.conv_dims = conv_dims
        self.m_dim = m_dim
        self.b_dim = b_dim
        self.num_classes = num_classes
        self.with_features = with_features
        self.f_dim = f_dim
        self.dropout_rate = dropout_rate

        self.activation_f = nn.Tanh()
        
        # Parse conv_dims
        graph_conv_dim, aux_dim, linear_dim = conv_dims
        
        # Graph convolution layers
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, with_features, f_dim, dropout_rate)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1] + m_dim, aux_dim, self.activation_f, with_features, f_dim, dropout_rate)
        self.multi_dense_layers = MultiDenseLayers(aux_dim, linear_dim, self.activation_f, dropout_rate)
        
        # Real/Fake output head (WGAN)
        self.output_layer = nn.Linear(linear_dim[-1], 1)
        
        # Auxiliary Classifier head (for LogP class prediction)
        self.class_layer = nn.Linear(linear_dim[-1], num_classes)

    def forward(self, adjacency_tensor, hidden, node, activation=None):
        """
        Forward pass through discriminator.
        
        Args:
            adjacency_tensor: Adjacency tensor of shape (batch, vertexes, vertexes, edges)
            hidden: Optional hidden features
            node: Node tensor of shape (batch, vertexes, m_dim)
            activation: Optional activation for real/fake output
        
        Returns:
            rf_output: Real/Fake logits of shape (batch, 1)
            class_logits: Class prediction logits of shape (batch, num_classes)
            features: Intermediate features of shape (batch, linear_dim[-1])
        """
        # Graph convolution
        adj = adjacency_tensor[:, :, :, 1:].permute(0, 3, 1, 2)
        h = self.gcn_layer(node, adj, hidden)
        h = self.agg_layer(node, h, hidden)
        features = self.multi_dense_layers(h)

        # Real/Fake output
        rf_output = self.output_layer(features)
        rf_output = activation(rf_output) if activation is not None else rf_output
        
        # Class prediction output (auxiliary classifier)
        class_logits = self.class_layer(features)

        return rf_output, class_logits, features
    
    def get_class_predictions(self, adjacency_tensor, hidden, node):
        """
        Get class predictions (probabilities) for molecules.
        
        Returns:
            class_probs: Class probabilities of shape (batch, num_classes)
        """
        _, class_logits, _ = self.forward(adjacency_tensor, hidden, node)
        return F.softmax(class_logits, dim=1)
