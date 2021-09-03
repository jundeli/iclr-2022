import os
import os.path as osp
import pickle
import numpy as np
import math

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import pennylane as qml
from sklearn.decomposition import PCA

training = pickle.load(open('dataset/training.pkl', 'rb'))
validation = pickle.load(open('dataset/validation.pkl', 'rb'))
data = set(training + validation)

for pdb_id in list(set([i.split('-')[0] for i in os.listdir('dataset/res')])):
    if not osp.isfile('dataset/graph/' + pdb_id + '-lig.npy'):
        data.discard(pdb_id)

N = 31
reduced_data = []
bond = np.zeros((36, 36))
atom = np.zeros((36))
for pdb_id in list(data):
    bond_one_hot, atom_one_hot = pickle.load(open('dataset/onehot/' + pdb_id + '-lig.pkl', 'rb'))
    # reverse onehot encoding
    for i in range(36):
        for j in range(36):
            bond[i, j] = np.argmax(bond_one_hot[i, j])
        atom[i] = np.argmax(atom_one_hot[i])

    # reduce maximum number of heavy atoms to N
    # and pad 32 zeros to form 1024=2^n_qubits dimension ligands
    if len(atom[atom > 0]) <= N:
        reduced_atom = atom[:N]
        reduced_bond = bond[:N, :N]
        reduced_data.append(np.concatenate((reduced_bond.reshape(-1), \
                            reduced_atom.reshape(-1), np.zeros((32))), axis=0))

n_samples = len(reduced_data)
train = np.array(reduced_data[:int(n_samples*0.85)])
test = np.array(reduced_data[int(n_samples*0.85):])

n_features = train.shape[-1]
n_qubits = int(math.log(n_features, 2))
latent_dim = 64
qml.enable_tape()
dev = qml.device("default.qubit.tf", wires=n_qubits)

@qml.qnode(dev, interface='tf', diff_method='backprop')
def qnode_e(inputs, weights):
    qml.templates.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize = True)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

@qml.qnode(dev, interface='tf', diff_method='backprop')
def qnode_d(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return qml.probs(wires=[i for i in range(n_qubits)])

weight_shapes_e = {"weights": (6, n_qubits, 3)}
weight_shapes_d = {"weights": (6, n_qubits, 3)}

qlayer_e = qml.qnn.KerasLayer(qnode_e, weight_shapes_e, output_dim=n_qubits)
qlayer_d = qml.qnn.KerasLayer(qnode_d, weight_shapes_d, output_dim=n_features)

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            qlayer_e,
            layers.Dense(n_qubits)
        ])
        
        self.decoder = tf.keras.Sequential([
            qlayer_d,
            layers.Dense(n_features)
        ])
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder(latent_dim)

opt = tf.keras.optimizers.Adam(learning_rate=0.1)
autoencoder.compile(optimizer=opt, loss=losses.MeanSquaredError())

autoencoder.fit(train, train,
                epochs=20,
                shuffle=True,
                validation_data=(test, test))