import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit
from qiskit.visualization.state_visualization import _bloch_multivector_data
from qiskit.visualization.bloch import Bloch
from qiskit_algorithms.optimizers import ADAM, COBYLA, SPSA

#### Test function colab ####
def test_function():
    print('Exercises ready!')
    
### DATASETS ###

def get_iris(seed):
    iris = datasets.load_iris()
    Y = iris.target[:100] 
    X = np.array([x / np.linalg.norm(x) for x in iris.data[:100]]) #Normalizing the data 
    nb_classes = 2 
    test_ratio = 0.2
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_ratio, random_state=seed, stratify=Y)

    return x_train,y_train,x_test,y_test


def get_non_seperable_data():   
    # 30 items in the dataset, each data point has a single feature
    dataset_size = 30
    nb_features = 1
    # Creating random values in the interval [-pi, pi]
    np.random.seed(4)
    x = (np.random.random_sample(dataset_size * nb_features).reshape(dataset_size, nb_features) * 2 - 1) * np.pi
    # The function that partitions our values into two classes.
    #   This is the function we want to learn!
    filter = (np.abs(x - 1) > np.pi/2)
    # Data points with label 0
    x0 = x[~filter]
    x0 = x0.reshape(len(x0), nb_features)
    # Data points with label 1
    x1 = x[filter]
    x1 = x1.reshape(len(x1), nb_features)
    # The labels
    y = filter.astype(int)

    return(x0,x1)

def get_seperable_data():
    dataset_size = 20 # 20 items in the dataset
    nb_features = 1 # Each data point has a single feature
    x = (np.random.random(dataset_size * nb_features).reshape(dataset_size, nb_features) * 2 - 1) * np.pi # Creating random values in the interval [-pi, pi]
    # The function that partitions our values into two classes.
    filter = x > 0
    # Data points with label 0
    x0 = x[~filter]
    x0 = x0.reshape(len(x0), nb_features)
    # Data points with label 1
    x1 = x[filter]
    x1 = x1.reshape(len(x1), nb_features)
    # The labels
    y = filter.astype(int)
    
    return(x,y,x0,x1)


## Embedding ##

def angle_embedding(x_params,nb_features):
       
    """
    Qubit - or rotation - encoding in RX gates.

    :param qc: The quantum circuit.
    :param nb_features: The number of features of the feature vector. 
    :return: The quantum circuit with the embedding layer. 
    """
    
    qc = QuantumCircuit(nb_features)

    for i in range(nb_features):
        qc.rx(x_params[i], i)

    return qc 


## LAB 3 ##

def get_statevector(qc, x, param_x, backend, params_var=None):
    """
    Binds the parameters of a parametrized quantum circuit to their values and execute
    the circuit with the statevector simulator.
    :param qc: The parametrized quantum circuit (PQC)
    :param x: The dataset containing all feature vectors (input data).
    :param param_x: The list of parameters present in the PQC that should be binded to
                    the feature vector. Both lenght must match.
    :param backend: The instance of the Statevector simulator.
    :param params_var: A dictionary of parameters with their respective values that
                       could have to be assigned in the PQC (other than parameters for
                       input data). Can be `None`.
    :return: A list of statevector, one for each input data.
    """
    grounded_circ = []
    for x_i in x:
        params_dict = {p:v for (p,v) in zip(param_x, x_i)}
        if params_var:
            params_dict.update(params_var)
        grounded_circ.append(qc.assign_parameters(params_dict))

    # result = execute(grounded_circ, backend).result()
    result = backend.run(grounded_circ).result()
    statevectors = [result.get_statevector(i) for i in range(len(grounded_circ))]

    return statevectors


def get_score(probs, targets):
    """
    Computes wether the predicted class (the one with highest probability)
    corresponds to the target.

    :param probs: A list of probability distributions.
    :param targets: The list of targets. The lenght of this list must match the
                    lenght of the list of probabilities.
    :return: A list of booleans telling if the predicted class corresponds to
             the targets.
    """
    predict = np.argmax(probs, axis=1)
    targets = np.array(targets).reshape(predict.shape)
    return predict == targets


def get_accuracy(score):
    """
    Compute the accuracy of the classification task.

    :param score: A list of booleans telling if the predicted class corresponds to
                  the targets.
    :return: The accuracy.
    """
    return np.sum(score) / len(score)


## Visualization tools ##

class BlochVisualization(Bloch):

    def __init__(self, fig=None, axes=None, view=None, figsize=None, background=False):
        super().__init__(fig, axes, view, figsize, background)
        self.pt_col = []
        self.pt_mark = []

    def add_points(self, points, meth='s', color='b', marker=None):
        super().add_points(points, meth)
        self.pt_col.append(color)
        self.pt_mark.append(marker)

    def plot_points(self):
        """Plot points"""
        # -X and Y data are switched for plotting purposes
        for k in range(len(self.points)):
            num = len(self.points[k][0])
            dist = [np.sqrt(self.points[k][0][j] ** 2 +
                            self.points[k][1][j] ** 2 +
                            self.points[k][2][j] ** 2) for j in range(num)]
            if any(abs(dist - dist[0]) / dist[0] > 1e-12):
                # combine arrays so that they can be sorted together
                zipped = list(zip(dist, range(num)))
                zipped.sort()  # sort rates from lowest to highest
                dist, indperm = zip(*zipped)
                indperm = np.array(indperm)
            else:
                indperm = np.arange(num)
            pnt_colors = np.array(self.point_color * int(np.ceil(num / float(len(self.point_color)))))

            pnt_colors = pnt_colors[0:num]
            pnt_colors = list(pnt_colors[indperm])
            marker = self.point_marker[np.mod(k, len(self.point_marker))]
            pnt_size = self.point_size[np.mod(k, len(self.point_size))]
            if self.pt_mark[k] is not None:
                marker = self.pt_mark[k]
            self.axes.scatter(np.real(self.points[k][1][indperm]), -np.real(self.points[k][0][indperm]),
                              np.real(self.points[k][2][indperm]), s=pnt_size, alpha=1, edgecolor=None,
                              zdir='z', color=self.pt_col[k], marker=marker)

    def plot_axes_labels(self):
        """axes labels"""
        opts = {'fontsize': self.font_size,
                'color': self.font_color,
                'horizontalalignment': 'center',
                'verticalalignment': 'center'}
        opts0 = {'fontsize': self.font_size,
                'color': 'b',
                'horizontalalignment': 'center',
                'verticalalignment': 'center'}
        opts1 = {'fontsize': self.font_size,
                'color': 'r',
                'horizontalalignment': 'center',
                'verticalalignment': 'center'}
        self.axes.text(0, -self.xlpos[0], 0, self.xlabel[0], **opts)
        self.axes.text(0, -self.xlpos[1], 0, self.xlabel[1], **opts)

        self.axes.text(self.ylpos[0], 0, 0, self.ylabel[0], **opts)
        self.axes.text(self.ylpos[1], 0, 0, self.ylabel[1], **opts)

        self.axes.text(0, 0, self.zlpos[0], self.zlabel[0], **opts0)
        self.axes.text(0, 0, self.zlpos[1], self.zlabel[1], **opts1)


def plot_bloch_visualization(statevectors, colors=['b', 'r'], x=None, x0=None, x1=None, score=None, fig=None, axes=None):
    """
    Utility function used to plot a list of statevector on the surface of the Bloch sphere.

    :param statevectors: A tuple of lists of statevectors. The first list contains statevectors 
                         associated to class 0 while the second list contains statevectors associated
                         to class 1.
    :param colors: List of colors to use for label 0 and 1 respectively. Default 0 is blue and 1 is red.
    :param x: The dataset containing all feature vectors (input data). Can be `None` if same markers are
              used for all statevectors.
    :param x0: The list of data point with label 0. Can be `None` if same markers are used for all statevectors.
    :param x1: The list of data point with label 1. Can be `None` if same markers are used for all statevectors.
    :param score: A list of booleans that specifies if the predicted class of each data points corresponds to
                  the target label. The lenght of this list should match the size of the dataset.
                  Can be `None` if same markers are used for all statevectors.
    """
    bloch = BlochVisualization(fig=fig, axes=axes)

    marker = 'o'
    for i, s in enumerate(statevectors[0]):
        if x0 is not None:
            bloch.point_size = [100]
            index = np.where((x == x0[i].item()).flatten())[0].item()
            marker = '.' if score[index] else 'x'
        bloch.add_points(_bloch_multivector_data(s)[0], meth='m', color=colors[0], marker=marker)

    marker = 'o'
    for i, s in enumerate(statevectors[1]):
        if x1 is not None:
            bloch.point_size = [100]
            index = np.where((x == x1[i].item()).flatten())[0].item()
            marker = '.' if score[index] else 'x'
        bloch.add_points(_bloch_multivector_data(s)[0], meth='m', color=colors[1], marker=marker)

    bloch.make_sphere()
