"""
Sidequest 3 — Section 7: Train a Quantum Binary Classifier on the Iris dataset
using qiskit-machine-learning. Goal: > 98% accuracy.

Pipeline:
- Embedding: ZZFeatureMap (more expressive than plain angle embedding)
- Variational ansatz: RealAmplitudes (light-weight, trainable)
- Interpretation: parity of the bitstring -> class in {0, 1}
- QNN: SamplerQNN
- Trainer: NeuralNetworkClassifier with COBYLA optimizer
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA

SEED = 8398
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def preprocessing(test_ratio: float, seed: int):
    iris = datasets.load_iris()
    Y = iris.target[:100]  # keep only classes 0 and 1
    X = np.array([x / np.linalg.norm(x) for x in iris.data[:100]])
    return train_test_split(X, Y, test_size=test_ratio, random_state=seed, stratify=Y)


nb_features = 4
nb_classes = 2
test_ratio = 0.2

x_train, x_test, y_train, y_test = preprocessing(test_ratio, SEED)
print(f"Train set: {len(x_train)} | Test set: {len(x_test)}")


# ---------------------------------------------------------------------------
# Quantum pipeline
# ---------------------------------------------------------------------------
# 1. Data embedding — ZZFeatureMap (richer than plain RX angle embedding)
emb_circuit = ZZFeatureMap(feature_dimension=nb_features, reps=2, entanglement="linear")

# 2. Trainable change-of-basis circuit — RealAmplitudes
ansatz = RealAmplitudes(num_qubits=nb_features, reps=2, entanglement="linear")

# Combine embedding + ansatz
qc = emb_circuit.compose(ansatz)


# 3. Interpretation function: parity of the measured bitstring
def parity(x: int) -> int:
    return bin(x).count("1") % 2


# 4. Build the QNN (SamplerQNN)
sampler_qnn = SamplerQNN(
    circuit=qc,
    input_params=emb_circuit.parameters,
    weight_params=ansatz.parameters,
    interpret=parity,
    output_shape=nb_classes,
)

# 5. Optimizer + initial weights
num_iter = 200
optimizer = COBYLA(maxiter=num_iter)
initial_weights = np.random.rand(ansatz.num_parameters)


# Optional: live callback to watch training
loss_history: list[float] = []


def callback(weights, obj_value):  # noqa: ARG001
    loss_history.append(float(obj_value))
    if len(loss_history) % 10 == 0:
        print(f"  iter {len(loss_history):3d} | loss = {obj_value:.4f}")


circuit_classifier = NeuralNetworkClassifier(
    neural_network=sampler_qnn,
    optimizer=optimizer,
    initial_point=initial_weights,
    callback=callback,
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
print("\nTraining quantum binary classifier ...")
circuit_classifier.fit(x_train, y_train)

train_acc = circuit_classifier.score(x_train, y_train)
test_acc = circuit_classifier.score(x_test, y_test)

print("\n=================== RESULTS ===================")
print(f"  Accuracy on the training set : {train_acc*100:.2f}%")
print(f"  Accuracy on the test set     : {test_acc*100:.2f}%")
print("===============================================")

if test_acc >= 0.98:
    print("Target reached: test accuracy >= 98% — sidequest validated.")
else:
    print("Target NOT reached — consider more iterations or a different ansatz/embedding.")


# ---------------------------------------------------------------------------
# Save loss curve
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title("Training loss — Quantum Binary Classifier")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./side_quests/challenge_03_qml_image_classification/loss_curve.png", dpi=120)
    print("Saved loss curve to loss_curve.png")
except Exception as exc:
    print(f"(plot skipped: {exc})")
