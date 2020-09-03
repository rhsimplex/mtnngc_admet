import collections

import numpy as np
import six
import tensorflow as tf

from deepchem.data import NumpyDataset, pad_features
from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
from deepchem.models.tensorgraph.graph_layers import WeaveGather, \
    DTNNEmbedding, DTNNStep, DTNNGather, DAGLayer, \
    DAGGather, DTNNExtract, MessagePassing, SetGather
from deepchem.models.tensorgraph import TensorGraph
from deepchem.models.tensorgraph.graph_layers import WeaveLayerFactory
from deepchem.models.tensorgraph.layers import Layer, Dense, SoftMax, Reshape, \
    SoftMaxCrossEntropy, GraphConv, BatchNorm, Exp, ReduceMean, ReduceSum, \
    GraphPool, GraphGather, WeightedError, Dropout, BatchNorm, Stack, Flatten, GraphCNN, GraphCNNPool, Divide
from deepchem.models.tensorgraph.layers import L2Loss, Label, Weights, Feature
from deepchem.trans import undo_transforms
from tensorflow.python.keras.regularizers import Regularizer


class TrimGraphOutput(Layer):
    """Trim the output to the correct number of samples.

    GraphGather always outputs fixed size batches.  This layer trims the output
    to the number of samples that were in the actual input tensors.
    """

    def __init__(self, in_layers, **kwargs):
        super(TrimGraphOutput, self).__init__(in_layers, **kwargs)
        try:
            s = list(self.in_layers[0].shape)
            s[0] = None
            self._shape = tuple(s)
        except:
            pass

    def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
        inputs = self._get_input_tensors(in_layers)
        n_samples = tf.shape(inputs[1])[0]
        out_tensor = inputs[0][0:n_samples]
        if set_tensors:
            self.out_tensor = out_tensor
        return out_tensor


def gini_decay(factor):
    variables = []
    for v in tf.trainable_variables():
        # only pick up kernels the dense layers (in this case there should be only one, since no_fcn should be true)
        if v.get_shape().ndims == 2 and v.name.startswith('Dense'):
            variables.append(v)

    x = v[0]
    with tf.name_scope('weight_decay'):
        # flatten the kernal weight matrix
        d = tf.expand_dims(tf.reshape(x, [-1]), 0)
        # compute the nominator of the quantity here: https://en.wikipedia.org/wiki/Gini_coefficient#Definition
        # this always seems to come out to zero, and I don't know why. If you remove the additive factor, you will not be able to train as your loss will always be nans
        nominator = tf.reduce_sum(tf.abs(d - tf.transpose(d))) + 0.1  # <- remove this additive factor to reproduce
        denominator = 2. * ((512*10)**2 - (512*10)) * tf.reduce_mean(tf.abs(d)) + 0.0001  # small additive factor for numerical stability in the denominator
    return tf.truediv(nominator,  denominator) ** factor


class GiniDecay(Layer):
    """Apply gini coefficent adjustment to loss

    The Gini coefficient always ranges between 0 and 1. We can raise it to a factor to make the effect
    more extreme (gini ** factor), since this value will also lie between zero and one.

    The final loss is computed as final_loss = weighted_L2_loss / (gini ** factor)
    """

    def __init__(self, factor=10., **kwargs):
        """Create a weight decay penalty layer.

        Parameters
        ----------
        factor: float
          magnitude of the factor term
        """
        self.factor = factor
        super(GiniDecay, self).__init__(**kwargs)
        try:
            self._shape = tuple(self.in_layers[0].shape)
        except:
            pass

    def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
        inputs = self._get_input_tensors(in_layers)
        parent_tensor = inputs[0]
        # we want highly "inequal" kernels, so we divide by the gini coefficient, which is near one when the weights are inequal
        out_tensor = parent_tensor / gini_decay(self.factor)
        if set_tensors:
            self.out_tensor = out_tensor
        return out_tensor


class GraphConvModel(TensorGraph):

    def __init__(self,
                 n_tasks,
                 graph_conv_layers=[64, 64],
                 dense_layer_size=128,
                 dropout=0.0,
                 mode="classification",
                 number_atom_features=75,
                 n_classes=2,
                 uncertainty=False,
                 no_fcn=False,
                 gini_factor=0.,
                 **kwargs):
        """
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    graph_conv_layers: list of int
      Width of channels for the Graph Convolution Layers
    dense_layer_size: int
      Width of channels for Atom Level Dense Layer before GraphPool
    dropout: list or float
      the dropout probablity to use for each layer.  The length of this list should equal
      len(graph_conv_layers)+1 (one value for each convolution layer, and one for the
      dense layer).  Alternatively this may be a single value instead of a list, in which
      case the same value is used for every layer.
    mode: str
      Either "classification" or "regression"
    number_atom_features: int
        75 is the default number of atom features created, but
        this can vary if various options are passed to the
        function atom_features in graph_features
    n_classes: int
      the number of classes to predict (only used in classification mode)
    uncertainty: bool
      if True, include extra outputs and loss terms to enable the uncertainty
      in outputs to be predicted
    """
        if mode not in ['classification', 'regression']:
            raise ValueError("mode must be either 'classification' or 'regression'")
        self.n_tasks = n_tasks
        self.mode = mode
        self.dense_layer_size = dense_layer_size
        self.graph_conv_layers = graph_conv_layers
        kwargs['use_queue'] = False
        self.number_atom_features = number_atom_features
        self.n_classes = n_classes
        self.uncertainty = uncertainty
        self.no_fcn = no_fcn
        self.gini_factor = gini_factor
        if not isinstance(dropout, collections.Sequence):
            dropout = [dropout] * (len(graph_conv_layers) + 1)
        if len(dropout) != len(graph_conv_layers) + 1:
            raise ValueError('Wrong number of dropout probabilities provided')
        self.dropout = dropout
        if uncertainty:
            if mode != "regression":
                raise ValueError("Uncertainty is only supported in regression mode")
            if any(d == 0.0 for d in dropout):
                raise ValueError(
                    'Dropout must be included in every layer to predict uncertainty')
        super(GraphConvModel, self).__init__(**kwargs)
        self.build_graph()

    def build_graph(self):
        """
    Building graph structures:
    """
        self.atom_features = Feature(shape=(None, self.number_atom_features))
        self.degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
        self.membership = Feature(shape=(None,), dtype=tf.int32)

        self.deg_adjs = []
        for i in range(0, 10 + 1):
            deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
            self.deg_adjs.append(deg_adj)
        in_layer = self.atom_features
        for layer_size, dropout in zip(self.graph_conv_layers, self.dropout):
            gc1_in = [in_layer, self.degree_slice, self.membership] + self.deg_adjs
            gc1 = GraphConv(layer_size, activation_fn=tf.nn.relu, in_layers=gc1_in)
            batch_norm1 = BatchNorm(in_layers=[gc1])
            if dropout > 0.0:
                batch_norm1 = Dropout(dropout, in_layers=batch_norm1)
            gp_in = [batch_norm1, self.degree_slice, self.membership] + self.deg_adjs
            in_layer = GraphPool(in_layers=gp_in)
        if not self.no_fcn:
            dense = Dense(
                out_channels=self.dense_layer_size,
                activation_fn=tf.nn.relu,
                in_layers=[in_layer])
            batch_norm3 = BatchNorm(in_layers=[dense])
            if self.dropout[-1] > 0.0:
                batch_norm3 = Dropout(self.dropout[-1], in_layers=batch_norm3)
            readout = GraphGather(
                batch_size=self.batch_size,
                activation_fn=tf.nn.tanh,
                in_layers=[batch_norm3, self.degree_slice, self.membership] +
                          self.deg_adjs)
        else:
            readout = GraphGather(
                batch_size=self.batch_size,
                activation_fn=tf.nn.tanh,
                in_layers=[in_layer, self.degree_slice, self.membership] +
                          self.deg_adjs)

        n_tasks = self.n_tasks
        weights = Weights(shape=(None, n_tasks))
        if self.mode == 'classification':
            n_classes = self.n_classes
            labels = Label(shape=(None, n_tasks, n_classes))
            logits = Reshape(
                shape=(None, n_tasks, n_classes),
                in_layers=[
                    Dense(in_layers=readout, out_channels=n_tasks * n_classes)
                ])
            logits = TrimGraphOutput([logits, weights])
            output = SoftMax(logits)
            self.add_output(output)
            loss = SoftMaxCrossEntropy(in_layers=[labels, logits])
            weighted_loss = WeightedError(in_layers=[loss, weights])
            self.set_loss(weighted_loss)
        else:
            labels = Label(shape=(None, n_tasks))
            regressor = Dense(in_layers=readout, out_channels=n_tasks)
            output = Reshape(
                shape=(None, n_tasks),
                in_layers=[regressor])
            output = TrimGraphOutput([output, weights])
            self.add_output(output)
            if self.uncertainty:
                log_var = Reshape(
                    shape=(None, n_tasks),
                    in_layers=[Dense(in_layers=readout, out_channels=n_tasks)])
                log_var = TrimGraphOutput([log_var, weights])
                var = Exp(log_var)
                self.add_variance(var)
                diff = labels - output
                weighted_loss = weights * (diff * diff / var + log_var)
                weighted_loss = ReduceSum(ReduceMean(weighted_loss, axis=[1]))
            else:
                weighted_loss_no_decay = ReduceSum(L2Loss(in_layers=[labels, output, weights]))
                if self.gini_factor > 0.:
                    weighted_loss = GiniDecay(self.gini_factor, in_layers=weighted_loss_no_decay)
                else:
                    weighted_loss = weighted_loss_no_decay
            self.set_loss(weighted_loss)

    def default_generator(self,
                          dataset,
                          epochs=1,
                          predict=False,
                          deterministic=True,
                          pad_batches=True):
        for epoch in range(epochs):
            for ind, (X_b, y_b, w_b, ids_b) in enumerate(
                    dataset.iterbatches(
                        self.batch_size,
                        pad_batches=pad_batches,
                        deterministic=deterministic)):
                d = {}
                if self.mode == 'classification':
                    d[self.labels[0]] = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                        -1, self.n_tasks, self.n_classes)
                else:
                    d[self.labels[0]] = y_b
                d[self.task_weights[0]] = w_b
                multiConvMol = ConvMol.agglomerate_mols(X_b)
                d[self.atom_features] = multiConvMol.get_atom_features()
                d[self.degree_slice] = multiConvMol.deg_slice
                d[self.membership] = multiConvMol.membership
                for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
                    d[self.deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
                yield d

    def predict_on_smiles(self, smiles, transformers=[], untransform=False):
        """Generates predictions on a numpy array of smile strings

            # Returns:
              y_: numpy ndarray of shape (n_samples, n_tasks)
            """
        max_index = len(smiles) - 1
        n_tasks = len(self.outputs)
        num_batches = (max_index // self.batch_size) + 1
        featurizer = ConvMolFeaturizer()

        y_ = []
        for i in range(num_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, max_index + 1)
            smiles_batch = smiles[start:end]
            y_.append(
                self.predict_on_smiles_batch(smiles_batch, featurizer, transformers))
        y_ = np.concatenate(y_, axis=0)[:max_index + 1]
        y_ = y_.reshape(-1, n_tasks)

        if untransform:
            y_ = undo_transforms(y_, transformers)

        return y_


#################### Deprecation warnings for renamed TensorGraph models ####################

import warnings

TENSORGRAPH_DEPRECATION = "{} is deprecated and has been renamed to {} and will be removed in DeepChem 3.0."


class GraphConvTensorGraph(GraphConvModel):

    def __init__(self, *args, **kwargs):
        warnings.warn(
            TENSORGRAPH_DEPRECATION.format("GraphConvTensorGraph",
                                           "GraphConvModel"), FutureWarning)

        super(GraphConvTensorGraph, self).__init__(*args, **kwargs)
