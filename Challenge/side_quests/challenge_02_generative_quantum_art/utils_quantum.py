import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

def run_simulation(circuit, shots=1024, method='automatic'):
    """
    Run a quantum circuit on the Aer simulator.

    Args:
        circuit (QuantumCircuit): The circuit to run.
        shots (int): Number of shots. Default is 1024.
        method (str): Simulation method ('automatic', 'statevector', 'matrix_product_state', etc.).
                      'matrix_product_state' is recommended for larger grids (e.g. > 20 qubits).

    Returns:
        counts (dict): Dictionary of measurement counts.
    """
    simulator = AerSimulator(method=method)
    # Transpile the circuit for the simulator to ensure all gates (like mcx) are supported
    #transpiled_circuit = transpile(circuit, simulator)
    job = simulator.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return counts

def decode_bitstring_to_image(bitstring, shape):
    """
    Convert a measurement bitstring back to a 2D image array.
    Qiskit returns bitstrings in reverse order (qubit 0 is rightmost).

    Args:
        bitstring (str): The measured bitstring (e.g., '0110...').
        shape (tuple): The shape of the image (rows, cols).

    Returns:
        numpy.ndarray: The reconstructed binary image.
    """
    # Reverse the bitstring to get correct qubit order (qubit 0 -> index 0)
    bitstring_reversed = bitstring[::-1]
    
    # Convert to array of integers
    pixel_array = np.array([int(bit) for bit in bitstring_reversed])
    
    # Reshape to original image dimensions
    reconstructed_matrix = pixel_array.reshape(shape)
    
    return reconstructed_matrix

def reconstruct_intensity_from_counts(counts, shape, shots, scale=255.0):
    """
    Reconstruct a grayscale image from measurement counts by calculating
    the probability of measuring '1' for each pixel.

    General Idea (Steps):
    1. Concept: The grayscale intensity of a pixel is determined by the probability of measuring its corresponding qubit in state |1>.
    2. We analyze the dictionary of counts (e.g., {'101': 50, '111': 30, '010': 20}).
    3. For every qubit (pixel), we calculate how often it was measured as '1' across all outcomes.
       - Example: Take the rightmost qubit (index 0).
       - It is '1' in '101' (adds 50).
       - It is '1' in '111' (adds 30).
       - It is '0' in '010' (adds 0).
       - Total count for this qubit = 80.
    4. Calculate probability: P(1) = Total Count / Total Shots.
    5. Scale this probability to the image range (e.g., 0-255).

    Args:
        counts (dict): Measurement counts from the simulation.
        shape (tuple): The shape of the image (rows, cols).
        shots (int): Total number of shots used.
        scale (float): Scaling factor for the output (e.g., 255.0 for 0-255 range).

    Returns:
        numpy.ndarray: The reconstructed intensity image.
    """
    num_qubits = shape[0] * shape[1]
    ones_count = np.zeros(num_qubits)
    
    for bitstring, count in counts.items():
        # Reverse bitstring to match qubit indices
        bitstring_reversed = bitstring[::-1]
        for qubit_idx, bit in enumerate(bitstring_reversed):
            if bit == '1':
                ones_count[qubit_idx] += count
                
    # Calculate probabilities and scale
    probabilities = ones_count / shots
    reconstructed_matrix = probabilities.reshape(shape) * scale
    
    return reconstructed_matrix


def pixel_index(row, col, cols):
    """
    Calculate the linear qubit index for a pixel at (row, col).
    
    Why do we need this?
    Quantum circuits are 1-dimensional: qubits are indexed linearly (0, 1, 2, ...).
    Images are 2-dimensional: pixels have (row, column) coordinates.
    To represent an image on a quantum computer, we must map each 2D pixel to a unique 1D qubit.

    How it works (Row-Major Flattening):
    Imagine reading a book (left to right, top to bottom).
    1. We skip all the complete rows above our target pixel. Each row has 'cols' pixels.
       -> (row * cols)
    2. We add the position of the pixel within its current row.
       -> (+ col)

    Example:
    In a 4x4 image (cols=4):
    - Pixel (0, 0) -> 0 * 4 + 0 = Qubit 0
    - Pixel (0, 3) -> 0 * 4 + 3 = Qubit 3
    - Pixel (1, 0) -> 1 * 4 + 0 = Qubit 4 (Start of second row)

    Important Note:
    This logic applies specifically to the 'Qubit per Pixel' encoding strategy used in this project, 
    where each pixel is represented by a single qubit. Other encodings (like FRQI or NEQR) 
    use different mapping schemes.

    Args:
        row (int): Row index.
        col (int): Column index.
        cols (int): Number of columns in the image.
        
    Returns:
        int: Linear index (the specific qubit number corresponding to this pixel).
    """
    return row * cols + col


def visualize_quantum_results(counts, shape, num_examples=4, title="Quantum Pattern Results"):
    """
    Visualize the results of a quantum simulation by decoding bitstrings into images.
    
    Args:
        counts (dict): Dictionary of measurement counts from the simulation.
        shape (tuple): The shape of the image (rows, cols).
        num_examples (int): Number of examples to display. Default is 4.
        title (str): Title for the plot.
    """
    # Determine how many examples we can actually show (limited by available results or request)
    num_to_show = min(num_examples, len(counts))
    
    # Create a figure with subplots arranged in a row
    fig, axes = plt.subplots(1, num_to_show, figsize=(4 * num_to_show, 4))
    
    # If there's only one example, axes is not a list, so we wrap it
    if num_to_show == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    # Iterate through the first 'num_to_show' results from the counts dictionary
    # counts.items() returns pairs of (bitstring, count)
    # We convert to list and slice to get the first N examples
    for idx, (bitstring, count) in enumerate(list(counts.items())[:num_to_show]):
        
        # Decode the bitstring into a 2D image array
        # A bitstring represents a single measurement outcome (one 'shot') of the quantum circuit.
        # For example, '0011...' means qubit_n 0 measured 0, qubit_(n-1) 1 measured 0, etc. Note, Qiskit use little endian convention.
        # This function maps those bits back to pixel positions (row, col).
        image = decode_bitstring_to_image(bitstring, shape)
        
        # Display the image on the corresponding subplot
        axes[idx].imshow(image, cmap='gray_r', interpolation='nearest')
        axes[idx].set_title(f'Outcome #{idx+1}\n(Count: {count})')
        axes[idx].axis('off')
        
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
