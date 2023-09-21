import os
import re
import csv
import numpy as np
import pandas as pd
import pickle as pk
import networkx as nx
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from seaborn import set_palette, set_style

def sorted_alphanumeric(data):
    '''
    Sorts the given list of strings in an alphanumeric manner.

    Parameters:
        data (list): The list of strings to be sorted.

    Returns:
        list: The sorted list of strings.
    '''
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(data, key=alphanum_key)

def process_file(file_path):
    '''
    Process a GraphML file and return its adjacency matrix.

    Parameters:
        file_path (str): The path to the graphml file.

    Returns:
        numpy.ndarray: The adjacency matrix of the graph.
    '''
    data_flow = nx.read_graphml(file_path)
    adjacency_matrix = nx.to_numpy_array(data_flow)
    return adjacency_matrix

def RK4(func, y0, T, step):
    """
    Fourth Order Runge-Kutta method for numerically integrating a system of ordinary differential equations.

    This function computes the numerical solution of a system of ordinary differential equations using the Fourth 
    Order Runge-Kutta method (RK4). The RK4 method is a time-stepping algorithm that iteratively estimates the state 
    variables at each time point by solving the system of equations through a weighted average of derivatives.

    Args:
        func (callable): A function that computes the time derivative of the state variables at a given time point.
            The function should take two arguments: y (array-like) representing the current state variables, and t (float)
            representing the current time.
        y0 (array-like): The initial conditions of the system, a 1D array-like object containing the values of the state
            variables at the initial time point.
        T (array-like): The time grid, a 1D array-like object containing the time points at which the numerical integration
            should be performed.
        step (float): The step size for numerical integration. It represents the time step between consecutive time points
            in the time grid.

    Returns:
        y (array-like): A 2D array-like object containing the state variables at each time point in the time grid. The rows
            of the array represent the time steps, and the columns represent the state variables.
    """
    y = np.zeros((len(T), len(y0)))
    y[0] = y0
    dt = step

    for i in range(len(T) - 1):
        t = T[i]
        k1 = func(y[i], t) 
        k2 = func(y[i] + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = func(y[i] + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = func(y[i] + dt * k3, T[i + 1])
        y[i + 1] = y[i] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return y

def frobenius_norm(*args):
    '''
    Compute the Frobenius norm of one or two matrices.

    If one matrix is provided, it computes the Frobenius norm of that matrix.
    If two matrices are provided, it computes the Frobenius norm of their difference.

    The Frobenius norm is a measure of the difference between two matrices and can be computed by taking the square root
    of the sum of the squared differences of corresponding elements in the matrices.

    Parameters:
        *args: One or two numpy.ndarray matrices.
        
    Returns:
        float: The Frobenius norm.
    '''
    if len(args) == 1:
        return np.linalg.norm(args[0])
    else:
        return np.linalg.norm(args[0] - args[1])

def compare_trajectories(trajectories_path):
    """
    Compare trajectories using the Frobenius norm and Pearson Correlation.

    Parameters:
        trajectories_path (str): Path to the pickle file containing trajectories.

    Returns:
        dict: A dictionary containing comparison metrics for each trajectory compared to the baseline trajectory.
            The keys are the resolution labels, and the values are dictionaries with the following metrics:
            - 'frobenius_norm': Normalized Frobenius norm between trajectories.
            - 'correlation': Pearson correlation between trajectories.
            - 'p_value': P-value associated with the Pearson correlation.
    """
    comparisons = {}

    with open(trajectories_path, 'rb') as file:
        trajectories = pk.load(file)

    baseline_trajectory = trajectories['baseline']
    baseline_norm = np.linalg.norm(baseline_trajectory)

    for label, trajectory in trajectories.items():
        if label != 'baseline':
            norm = frobenius_norm(baseline_trajectory, trajectory)
            correlation, p_value = pearsonr(baseline_trajectory.flatten(), trajectory.flatten())

            comparisons[label] = {
                'frobenius_norm': norm,
                'correlation': f'{correlation:.6f}',
                'p_value': f'{p_value:.6f}'
            }

    return comparisons

def create_metrics_dataframe(comparisons):
    """
    Create a pandas DataFrame from the comparison metrics.

    Parameters:
        comparisons (dict): A dictionary containing comparison metrics for each trajectory compared to the baseline trajectory.
            The keys are the resolution labels, and the values are dictionaries with metrics.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the comparison metrics for each resolution.
    """
    data = []

    for label, metrics in comparisons.items():
        resolution_number = re.search(r'_(\d+)$', label).group(1)
        if resolution_number == '1':
            formatted_resolution = f"1 net calculated from flow averages."
        else:
            formatted_resolution = f"1 net every {resolution_number} days"

        data.append({
            'Resolution': formatted_resolution,
            'Frobenius Norm': metrics['frobenius_norm'],
            'Correlation': metrics['correlation'],
            'P-Value': metrics['p_value']
        })

    df = pd.DataFrame(data)
    df.index = range(1, len(df) + 1)
    return df

def generate_travel_rates(N, save_path, days):
    """
    Generates random NxN matrices of travel rates between N locations for multiple days,
    where each column represents the travel rates from one location to all other locations.

    Parameters
    ----------
    N : int
        Number of locations.
    save_path : str
        The path to the directory where the generated CSV files will be saved.
    days : int
        Number of sets of travel rates to generate.

    Returns
    -------
    inflow_matrices : list of numpy.ndarray
        A list of NxN matrices of inflow travel rates, where inflow_matrices[i] represents 
        the inflow travel rates for day i.
    outflow_matrices : list of numpy.ndarray
        A list of NxN matrices of outflow travel rates, where outflow_matrices[i] represents 
        the outflow travel rates for day i.
    """
    inflow_matrices = []
    outflow_matrices = []

    # If the folder does not exist, we create it
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for day in range(days):
        # Create an empty NxN matrix for inflow
        inflow_matrix = np.zeros((N, N))
        outflow_matrix = np.zeros((N, N))

        for j in range(N):
            # Generate N random numbers between 0 and 1 for inflow
            inflow_column = np.random.rand(N)
            # Set the jth element of the inflow column to 0
            inflow_column[j] = 0.0
            # Ensure that all elements in inflow column are positive
            inflow_column = np.abs(inflow_column)
            # Normalize the inflow column to ensure that the sum of the columns is equal to 1
            inflow_column = inflow_column / np.sum(inflow_column)
            # Set the inflow column in the inflow matrix
            inflow_matrix[:, j] = inflow_column

            # Generate N random numbers between 0 and 1 for outflow
            outflow_row = np.random.rand(N)
            # Set the jth element of the outflow row to 0
            outflow_row[j] = 0.0
            # Ensure that all elements in outflow row are positive
            outflow_row = np.abs(outflow_row)
            # Normalize the outflow row to ensure that the sum of the rows is equal to 1
            outflow_row = outflow_row / np.sum(outflow_row)
            # Set the outflow row in the outflow matrix
            outflow_matrix[j, :] = outflow_row

        inflow_matrices.append(inflow_matrix)
        outflow_matrices.append(outflow_matrix)

        # Save the numpy arrays to CSV files
        np.savetxt(os.path.join(save_path, f'inflow_travel_rates_day_{day + 1}.csv'), inflow_matrix, delimiter=',')
        np.savetxt(os.path.join(save_path, f'outflow_travel_rates_day_{day + 1}.csv'), outflow_matrix, delimiter=',')

    return inflow_matrices, outflow_matrices

def get_OD_matrices(folder_name):
    '''
    Process inflow and outflow graphml files in parallel and save their adjacency matrices as numpy arrays.

    Parameters:
        folder_name (str): The path to the folder containing the graphml files.

    Note:
        The function processes inflow and outflow graphml files in parallel and saves their adjacency matrices
        as numpy arrays in 'inflow.npy' and 'outflow.npy' respectively. The function does not return anything.
    '''
    inflow_data = []
    outflow_data = []

    files = sorted_alphanumeric(os.listdir(folder_name))

    pool = mp.Pool()  

    for file_name in files:
        if 'in' in file_name:
            # Inflow file
            file_path = os.path.join(folder_name, file_name)
            inflow_data.append(pool.apply_async(process_file, (file_path,)))

        elif 'out' in file_name:
            # Outflow file
            file_path = os.path.join(folder_name, file_name)
            outflow_data.append(pool.apply_async(process_file, (file_path,)))
    
    pool.close()
    pool.join()

    inflow_data = np.array([result.get() for result in inflow_data])
    outflow_data = np.array([result.get() for result in outflow_data])

    np.save('inflow.npy', inflow_data)
    np.save('outflow.npy', outflow_data)

def check_flows(travel_rates):
    """
    Calculate the deviation of inflow and outflow rates from 100.00% for each time step.

    Parameters:
    - travel_rates (tuple of numpy.ndarray): A tuple of two 3D numpy arrays representing inflow and outflow rates
        between subpopulations. Both numpy.ndarrays should have dimensions (T, K, K), where K is the number of subpopulations
        and T is 

    Returns:
    - inflow_check (numpy.ndarray): Array of absolute deviations of inflow rates from 100.00% for each time step.
        The shape of this array is (T, N), where T is the number of time steps and N is the number of subpopulations.
    - outflow_check (numpy.ndarray): Array of absolute deviations of outflow rates from 100.00% for each time step.
        The shape of this array is (T, N), where T is the number of time steps and N is the number of subpopulations.
    """
    inflow_check = [np.sum(i - np.eye(len(i)) * 100.0, axis=0) for i in travel_rates[0]]
    outflow_check = [np.sum(o - np.eye(len(o)) * 100.0, axis=1) for o in travel_rates[1]]

    return np.abs(np.array(inflow_check)), np.abs(np.array(outflow_check))

def find_filled_elements_matrix_columns(matrix):
    """
    Find the filled/non-zero elements in each column of a given matrix.

    Args:
        matrix (numpy.ndarray): The input matrix, represented as a 2D NumPy array.

    Returns:
        list:
            A list of integers, where each integer represents the count of filled elements in the corresponding column.
    """
    filled_elements_quantity = []

    for j in range(matrix.shape[1]):
        filled_count = 0
        for i in range(matrix.shape[0]):
            if matrix[i, j] != 0.00:
                filled_count += 1
        filled_elements_quantity.append(filled_count)

    return filled_elements_quantity

def find_filled_elements_matrix_rows(matrix):
    """
    Find the filled/non-zero elements in each row of a given matrix.

    Args:
        matrix (numpy.ndarray): The input matrix, represented as a 2D NumPy array.

    Returns:
        list: 
            A list of integers, where each integer represents the count of filled elements in the corresponding row.
    """
    filled_elements_quantity = []

    for i in range(matrix.shape[0]):
        # Find filled elements in each row
        filled_count = 0
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0.00:
                filled_count += 1
        filled_elements_quantity.append(filled_count)

    return filled_elements_quantity

def find_filled_elements(travel_rates):
    """
    Find and return the indices of the filled/non-zero elements of inflow and outflow matrices.

    Parameters:
    travel_rates (tuple of numpy.ndarray): A tuple containing two 3D NumPy arrays representing inflow and outflow matrices
                                            with dimensions (T, K, K).

    Returns:
    dict: A dictionary containing the following keys and corresponding values:
        - 'quantity_inflow_filled_elements' (tuple of lists with ints): An array containing the counts of filled elements in each column (inflow)
            of the respective matrices.
        - 'quantity_outflow_filled_elements' (tuple of lists with ints): An array containing the counts of filled eintslements in each row (outflow)
            of the respective matrices.
    """
    pool = mp.Pool(processes=mp.cpu_count())
        
    inflow_filled_elements_quantity = pool.map(find_filled_elements_matrix_columns, travel_rates[0])
    outflow_filled_elements_quantity = pool.map(find_filled_elements_matrix_rows, travel_rates[1])
        
    return {
        'quantity_inflow_filled_elements' : inflow_filled_elements_quantity,
        'quantity_outflow_filled_elements' : outflow_filled_elements_quantity
    }

def partial_flows(flows_check, filled_elements_flows_counts):
    """
    Calculate partial inflow and outflow values based on the deviations of inflow and outflow rates.

    Parameters:
    flows_check (tuple): A tuple containing two numpy.ndarray arrays:
        - inflow_check (numpy.ndarray): Array of absolute deviations of inflow rates from 100.00% for each time step.
            The shape of this array is (T, N), where T is the number of time steps and N is the number of subpopulations.
        - outflow_check (numpy.ndarray): Array of absolute deviations of outflow rates from 100.00% for each time step.
            The shape of this array is (T, N), where T is the number of time steps and N is the number of subpopulations.

    filled_elements_flows_counts (tuple): A tuple containing two lists of integers:
        - filled_elements_inflow_counts (list of lists of ints): An array containing the counts of filled elements in each column (inflow)
            of the respective matrices. The outer list has a length of T (number of time steps), and the inner lists have a length of N (number of subpopulations).
        - filled_elements_outflow_counts (list of lists of ints): An array containing the counts of filled elements in each row (outflow)
            of the respective matrices. The outer list has a length of T (number of time steps), and the inner lists have a length of N (number of subpopulations).

    Returns:
    tuple: A tuple containing two numpy.ndarray arrays:
        - partial_inflow (numpy.ndarray): A 2D array of partial inflow values calculated for each element that is filled in the inflow matrices. 
                                            The shape of this array is (T, N), where T is the number of time steps and N is the number of subpopulations.
        - partial_outflow (numpy.ndarray): A 2D array of partial outflow values calculated for each element that is filled in the outflow matrices. 
                                            The shape of this array is (T, N), where T is the number of time steps and N is the number of subpopulations.
    """
    inflow_check, outflow_check = flows_check
    filled_elements_inflow_counts, filled_elements_outflow_counts = filled_elements_flows_counts

    T, N = inflow_check.shape
    partial_inflow = np.empty((T, N))
    partial_outflow = np.empty((T, N))

    for t in range(T):
        for i in range(N):
            if filled_elements_inflow_counts[t][i] != 0:
                partial_inflow[t, i] = inflow_check[t, i] / filled_elements_inflow_counts[t][i]
            else:
                partial_inflow[t, i] = 0.0

            if filled_elements_outflow_counts[t][i] != 0:
                partial_outflow[t, i] = outflow_check[t, i] / filled_elements_outflow_counts[t][i]
            else:
                partial_outflow[t, i] = 0.0

    return partial_inflow, partial_outflow

def insert_partial_flows(travel_rates, partial_flows):
    """
    Insert partial inflow and outflow values into the specified positions in the input and output arrays.

    Parameters:
    - travel_rates (tuple of numpy.ndarray): A tuple containing two 3D numpy.ndarray arrays representing inflow and outflow rates
        between subpopulations. Both numpy.ndarray arrays should have dimensions (T, K, K), where T is the number of time steps,
        and K is the number of subpopulations.
    - partial_flows (tuple): A tuple containing two numpy.ndarray arrays:
        - partial_inflow (numpy.ndarray): A 2D array of partial inflow values calculated for each element that is filled of the inflow matrices. 
                                            The shape of this array is (T, N), where T is the number of time steps and N is the number of subpopulations.
        - partial_outflow (numpy.ndarray): A 2D array of partial outflow values calculated for each element that is filled of the outflow matrices. 
                                            The shape of this array is (T, N), where T is the number of time steps and N is the number of subpopulations.

    Returns:
    - normalized_flows (tuple of numpy.ndarray): A tuple containing the modified inflow and outflow arrays with partial flow values inserted.
    """
    inflow, outflow = travel_rates
    in_shape, out_shape = inflow.shape, outflow.shape
    partial_inflow, partial_outflow = partial_flows

    # Copy the original inflow and outflow arrays
    normalized_inflow = inflow.copy()
    normalized_outflow = outflow.copy()

    # Fill the filled elements in inflow with partial_inflow column by column
    for t in range(in_shape[0]):
        for col in range(in_shape[1]):
            for row in range(in_shape[2]):
                if normalized_inflow[t, row, col] != 0.0:
                    normalized_inflow[t, row, col] += partial_inflow[t, col]

    # Fill the filled elements in outflow with partial_outflow row by row
    for t in range(out_shape[0]):
        for row in range(out_shape[2]):
            for col in range(out_shape[1]):
                if normalized_outflow[t, row, col] != 0.0:
                    normalized_outflow[t, row, col] += partial_outflow[t, row]

    return (normalized_inflow, normalized_outflow)

def normalize_travel_rates(travel_rates):
    """
    Normalize inflow and outflow matrices by distributing missing flow based on deviations and filled elements.

    Parameters:
    - travel_rates (tuple of numpy.ndarray): A tuple of two 3D numpy arrays representing inflow and outflow rates
        between subpopulations. Both numpy.ndarrays should have dimensions (T, K, K), where K is the number of subpopulations
        and T is the number of time steps.

    Returns:
    tuple of numpy.ndarray: A tuple containing two 3D numpy arrays representing the normalized inflow and outflow matrices.
        The shape of each array is (T, K, K), where T is the number of time steps and K is the number of subpopulations.

    Description:
    This function normalizes the inflow and outflow matrices by distributing missing flow. It calculates the missing inflow and outflow
    based on the deviations of inflow and outflow rates from 100.00% and the filled elements. The normalized matrices are returned as a
    tuple of numpy.ndarrays.

    The normalization process involves filling filled/non-zero elements in the inflow and outflow matrices with partial values calculated based on 
    deviations and the count of filled elements. The filling is done row by row for outflow and column by column for inflow matrices.

    The function also saves the resulting normalized flows in a binary file named 'normalized_flows.pkl'.
    """
    inflow, outflow = travel_rates

    # Find filled elements per row (outflow) / column (inflow)
    status = find_filled_elements(travel_rates)
    filled_elements_inflow_counts = status['quantity_inflow_filled_elements']
    filled_elements_outflow_counts = status['quantity_outflow_filled_elements']

    # Calculate partial inflow and outflow values based on deviations and filled elements
    partial_inflow, partial_outflow = partial_flows(
        check_flows(travel_rates), 
        (filled_elements_inflow_counts, filled_elements_outflow_counts)
    )

    normalized_flows = insert_partial_flows(
        (inflow, outflow),
        (partial_inflow, partial_outflow)
    )

    with open('normalized_flows.pkl', 'wb') as file:
        pk.dump(normalized_flows, file)

    return normalized_flows

def identify_day_beginnings(time_grid, total_days, aggregated, aggregation_factor):
    """
    Identifies the indices in the time grid that represent the beginning of each day.

    Parameters:
        time_grid (numpy.ndarray): The array of time grid.
        total_days (int): The total number of days.
        aggregated (bool): Flag indicating if aggregated matrices are used.
        aggregation_factor (int): The aggregation factor.

    Returns:
        numpy.ndarray: An array of indices representing the beginning of each day.
    """
    if not aggregated:
        return np.array([int(i * (len(time_grid) / total_days)) for i in range(total_days)])
    else:
        original_resolution = len(time_grid) / total_days
        points_per_aggregated_day = int(original_resolution * aggregation_factor)
        day_beginnings = np.arange(0, len(time_grid) - points_per_aggregated_day + 1, points_per_aggregated_day)
        return day_beginnings

def capture_elements_per_day(time_grid, day_beginnings, aggregated):
    """
    Captures the elements in the interval of each day based on the day beginnings.

    Parameters:
        time_grid (numpy.ndarray): The array of time grid.
        day_beginnings (numpy.ndarray): An array of indices representing the beginning of each day.
        aggregated (bool): Flag indicating if aggregated matrices are used.

    Returns:
        numpy.ndarray: An array of elements in the interval of each day.
    """
    if not aggregated:
        day_elements = [time_grid[start : end] for start, end in zip(day_beginnings, day_beginnings[1:])]
        day_elements.append(time_grid[day_beginnings[-1]:])
        return np.array(day_elements, dtype=object)
    else:
        day_elements = []
        points_per_day = int(len(time_grid) / len(day_beginnings))
        remaining_points = len(time_grid)
        for _ in day_beginnings[:-1]:
            end = min(points_per_day, remaining_points)
            day_elements.append(time_grid[len(time_grid) - remaining_points : len(time_grid) - remaining_points + end])
            remaining_points -= end
        day_elements.append(time_grid[len(time_grid) - remaining_points:])
        return np.array(day_elements, dtype=object)

def temporal_verifier(arrays, value, dt):
    """
    Checks the index of the array (day) containing the given value (flow).

    Parameters:
        arrays (numpy.ndarray): Array of arrays.
        value: The value to search for.
        dt: The time step size.

    Returns:
        int: Index of the array containing the value, or -1 if not found.
    """
    arrays_flat = np.concatenate(arrays)

    # Check if the value is beyond the last time step, return the last index in such cases
    if value > arrays_flat[-1]:
        return len(arrays) - 1
            
    indices = np.where(np.abs(arrays_flat - value) < dt / 2.0)[0]
            
    # in case the timestep is not found, it means it is probably an intermediate value, due to RK4 midpoints computations.
    if indices.size == 0:
        # in this case, one should use (t - dt) / 2 to capture the previous time step.
        indices = np.where(np.abs(arrays_flat - (value - dt / 2.0)) < dt / 2.0)[0]

    index = indices[0]
    array_index = np.searchsorted(np.cumsum([len(arr) for arr in arrays]), index, side='right')
            
    return array_index

class FluxModel:
    """
    A class for simulating the movement of individuals between subpopulations over time, using the Eulerian
    approach. The model is based on a flux matrix that describes the rates of movement between subpopulations,
    and assumes that the population sizes within each subpopulation are continuous and homogeneous.

    Parameters
    ----------
    travel_rates : tuple(numpy.ndarray, numpy.ndarray)
        A tuple containing two 3D numpy ndarrays representing the rates of movement between subpopulations.
        The first numpy.ndarray is the inflow rates, and the second numpy.ndarray is the outflow rates.
        The shape of each ndarray is T (time/days) x K (number of nodes/subpopulations) x K. The (i,j)-th element
        of the inflow rates numpy.ndarray denotes the rate of movement from subpopulation i to subpopulation j.
        The (i,j)-th element of the outflow rates numpy.ndarray denotes the rate of movement from subpopulation j
        to subpopulation i. The diagonal of both numpy.ndarrays should be equal to 0.0.
    T : tuple(numpy.ndarray, float)
        A tuple containing the time grid and step size. The first element is a 1D numpy array specifying the time points
        at which the simulation should be run. The second element is a float representing the step size between time points.
    save_path : str
        The path to save the simulation results.
    aggregated : bool
        Flag indicating if aggregated matrices are used.
    aggregation_factor : int
        The aggregation factor.

    Attributes
    ----------
    travel_rates : tuple(numpy.ndarray, numpy.ndarray)
        The tuple of inflow and outflow numpy.ndarrays that specifies the rates of movement between subpopulations
        for each day of the temporal network.
    K : int
        The number of subpopulations.
    N0 : numpy.ndarray
        A 1D array that specifies the initial population sizes for each subpopulation. By default, it is set
        to an array of ones with length K, multiplied by 500.
    N : float
        The total population size at the beginning of the simulation.
    N_i : numpy.ndarray or None
        An TxN matrix, where N is the number of subpopulations and T is the number of time steps, that
        specifies the population size of each subpopulation at each time step. It is set to None initially.

    Methods
    -------
    __init__(self, travel_rates, T, save_path, aggregated, aggregation_factor)
        Initializes a new instance of the FluxModel class with the given parameters, and runs the simulation.
    simulate(self)
        Runs the simulation and returns the population sizes at each time step.
    plot_diffusion(self)
        Plots the diffusion of individuals between subpopulations over time.
    """
    def __init__(self, travel_rates, T, save_path, aggregated, aggregation_factor):
        """
        Initializes a new instance of the FluxModel class with the given parameters, and runs the simulation.

        Args:
            travel_rates (tuple(numpy.ndarray, numpy.ndarray)): A tuple containing two 3D numpy ndarrays representing
                the rates of movement between subpopulations. The first numpy.ndarray is the inflow rates, and the
                second numpy.ndarray is the outflow rates. The shape of each ndarray is T (time/days) x K
                (number of nodes/subpopulations) x K. The (i,j)-th element of the inflow rates numpy.ndarray
                denotes the rate of movement from subpopulation i to subpopulation j. The (i,j)-th element of
                the outflow rates numpy.ndarray denotes the rate of movement from subpopulation j to subpopulation i.
                The diagonal of both numpy.ndarrays should be equal to 0.0.
            T (tuple(numpy.ndarray, float)): A tuple containing the time grid and step size. The first element is a 1D
                numpy array specifying the time points at which the simulation should be run. The second element is
                a float representing the step size between time points.
            aggregated (bool): Flag indicating if aggregated matrices are used.
            aggregation_factor (int): The aggregation factor.
        """
        self.inflow_rates, self.outflow_rates = travel_rates
        self.aggregated = aggregated
        self.aggregation_factor = aggregation_factor
        self.T, self.step = T
        self.total_days = int(self.T[-1])
        self.save_path = save_path
        self.K = len(self.inflow_rates[0])
        self.N0 = np.ones(self.K) * 500    
        self.N = np.sum(self.N0)
        self.N_i = None
        self.day_beginnings = identify_day_beginnings(self.T, self.total_days, self.aggregated, self.aggregation_factor)
        self.day_elements = capture_elements_per_day(self.T, self.day_beginnings, self.aggregated)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.simulate() 
        self.plot_diffusion()

    def _serial_ode(self, N, t):
        """
        Computes the time derivative of the number of hosts in each metapopulation
        based on the Flux model, which describes hosts as diffusing from one
        metapopulation to another.

        Args:
        N (array-like): Array containing the current number of hosts in each
                        metapopulation.
        t (float): Current time point.

        Returns:
        dN_dt (array-like): Array containing the time derivative of the number of
                            hosts in each metapopulation.
        """
        if self.aggregation_factor != 60:
            print(f'FLUX MODEL -- DAY: {temporal_verifier(self.day_elements, t, self.step) + 1}')
            outflow = self.outflow_rates[temporal_verifier(self.day_elements, t, self.step)]
            inflow = self.inflow_rates[temporal_verifier(self.day_elements, t, self.step)]
        else:
            print(f'FLUX MODEL - WORST RESOLUTION!')
            outflow = self.outflow_rates
            inflow = self.inflow_rates

        dN_dt = np.zeros(self.K)
        for i in range(self.K):
            outgoing_flux = 0.0
            incoming_flux = 0.0
            for j in range(self.K):
                outgoing_flux += outflow[i, j] * N[i]
                incoming_flux += inflow[j, i] * N[j]
            dN_dt[i] = incoming_flux - outgoing_flux
        return dN_dt

    def simulate(self):
        '''
        Simulates the movement of individuals between subpopulations over time, using the Eulerian approach. 
        Returns a 2D array of the population sizes of each subpopulation at each time point.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (len(T), K) that specifies the population sizes of each subpopulation at each 
            time point.
        '''
        def save_integration_csv(self):
            if not self.aggregated:
                dir = os.path.join(self.save_path, f"FluxModel_integration_baseline")

                if not os.path.exists(dir):
                    os.makedirs(dir)

                with open(dir + '/FluxModel_integration_baseline.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    header = ['Subpopulation {}'.format(i + 1) for i in range(self.K)]
                    writer.writerow(header)
                    for i in range(len(self.N_i)):
                        writer.writerow(self.N_i[i])
            else:
                dir = os.path.join(self.save_path, f"FluxModel_integration_aggregated")

                if not os.path.exists(dir):
                    os.makedirs(dir)

                with open(dir + f'/FluxModel_integration_aggregated_{60 // self.aggregation_factor}.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    header = ['Subpopulation {}'.format(i + 1) for i in range(self.K)]
                    writer.writerow(header)
                    for i in range(len(self.N_i)):
                        writer.writerow(self.N_i[i])

        self.N_i = RK4(self._serial_ode, self.N0, self.T, self.step)
        save_integration_csv(self)
    
    def plot_diffusion(self):
        '''
        Plots the population sizes of each subpopulation over time, and saves the figure to a file.
        
        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object that contains the plot.
        '''
        set_palette('OrRd')
        fig = plt.figure(figsize=(20, 8), facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

        for i in range(self.K):
            ax.plot(self.T, self.N_i[:, i], '--', lw=2)

        ax.set_title('Diffusion of hosts among subpopulations', fontsize=22)
        ax.set_xlabel('Time', fontsize=18)
        ax.set_ylabel(r'$\frac{dN_{i}}{dt}$', fontsize=18)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
        #legend = ax.legend()
        #legend.get_frame().set_alpha(.7)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)

        if not self.aggregated:
            dir = os.path.join(self.save_path, f"FluxModel_integration_baseline")
            fig.savefig(os.path.join(dir, f"FluxModel_integration_baseline.pdf"))
        else:
            dir = os.path.join(self.save_path, f"FluxModel_integration_aggregated")
            fig.savefig(os.path.join(dir, f"FluxModel_integration_aggregated_{60 // self.aggregation_factor}.pdf"))

        plt.close(fig)

class SIRModel(FluxModel):
    """
    A class for simulating the Susceptible-Infected-Recovered (SIR) model with host diffusion among metapopulations,
    using the Eulerian approach. The model is based on a flux matrix that describes the rates of movement between
    subpopulations, and assumes that the population sizes within each subpopulation are continuous and homogeneous.

    Parameters
    ----------
    travel_rates : tuple(numpy.ndarray, numpy.ndarray)
        A tuple containing two 3D numpy ndarrays representing the rates of movement between subpopulations.
        The first numpy.ndarray is the inflow rates, and the second numpy.ndarray is the outflow rates.
        The shape of each ndarray is T (time/days) x K (number of nodes/subpopulations) x K. The (i,j)-th element
        of the inflow rates numpy.ndarray denotes the rate of movement from subpopulation i to subpopulation j.
        The (i,j)-th element of the outflow rates numpy.ndarray denotes the rate of movement from subpopulation j
        to subpopulation i. The diagonal of both numpy.ndarrays should be equal to 0.0.
    T : tuple(numpy.ndarray, float) 
        A tuple containing the time grid and step size. The first element is a 1D
        numpy array specifying the time points at which the simulation should be run. The second element is
        a float representing the step size between time points.
    save_path : str
        The path to save the simulation results.
    beta : numpy.ndarray
        A 1D array with length N that specifies the transmission rate for each subpopulation.
    gamma : float
        The recovery rate for the infection.
    aggregated : bool
        Flag indicating if aggregated matrices are used.
    aggregation_factor : int
        The aggregation factor.

    Attributes
    ----------
    beta : numpy.ndarray
        The transmission rate for each subpopulation.
    gamma : float
        The recovery rate for the infection.
    travel_rates : tuple
        Tuple of numpy.ndarray representing the inflow and outflow rates.
    K : int
        The number of subpopulations.
    N0 : numpy.ndarray
        A 1D array that specifies the initial population sizes for each subpopulation. By default, it is set
        to an array of ones with length K, multiplied by 500.
    Y0 : numpy.ndarray
        A 1D array that specifies the initial conditions for the differential equations.
    Y : numpy.ndarray or None
        An Tx3N matrix, where N is the number of subpopulations, T is the number of time steps, and 3 represents the
        number of variables in the SIR model (susceptible, infected, recovered). It stores the population size of each
        subpopulation at each time step. It is set to None initially.
    basic_reproduction_number : float
        The basic reproduction number of the SIR model.
    """
    def __init__(self, travel_rates, T, save_path, beta, gamma, aggregated=False, aggregation_factor=None):
        """
        Initializes a new instance of the SIRModel class with the given parameters, and runs the simulation.

        Args:
            travel_rates (Tuple): Tuple of numpy.ndarray representing the inflow and outflow rates for each day of simulation.
            T (tuple): Time grid and the step between the points.
            save_path (str): The path to save the simulation results.
            beta (numpy.ndarray): A 1D array with length N that specifies the transmission rate for each subpopulation.
            gamma (float): The recovery rate for the infection.
            aggregated (bool): Flag indicating if aggregated matrices are used.
            aggregation_factor (int): The aggregation factor.
        """
        super().__init__(travel_rates, T, save_path, aggregated, aggregation_factor)
        self.beta = np.asarray(beta)
        self.gamma = gamma
        self.Y0 = np.hstack((self.N0 - 100, self.N0 - 400, np.zeros(self.K)))
        self.Y = None
        self.basic_reproduction_number = self.R_0()
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.run_simulation()
        self.plot_system_ode()
    
    def _serial_system_ode(self, y, t):
        """
        Computes the system of ordinary differential equations for the SIR model.

        Parameters
        ----------
        y : numpy.ndarray
            A 1D array with length 3K that specifies the initial values of the populations for each subpopulation.
            The first K elements of the array correspond to the initial values of the susceptible population for each subpopulation,
            the next K elements correspond to the initial values of the infected population for each subpopulation,
            and the last K elements correspond to the initial values of the recovered population for each subpopulation.
        t : float
            Current time point.

        Returns
        -------
        numpy.ndarray
            A 1D array that specifies the values of the derivatives of the populations at each time point.
        """
        if self.aggregation_factor != 60:
            print(f'SIR MODEL -- DAY: {temporal_verifier(self.day_elements, t, self.step) + 1}')
            outflow = self.outflow_rates[temporal_verifier(self.day_elements, t, self.step)]
            inflow = self.inflow_rates[temporal_verifier(self.day_elements, t, self.step)]
        else:
            print('SIR MODEL - WORST RESOLUTION!')
            outflow = self.outflow_rates
            inflow = self.inflow_rates
        
        S = y[ : self.K]
        I = y[self.K : 2 * self.K]
        R = y[2 * self.K :]

        dS_dt = np.zeros(self.K)
        dI_dt = np.zeros(self.K)
        dR_dt = np.zeros(self.K)

        for i in range(self.K):
            for j in range(self.K):
                dS_dt[i] -= outflow[i, j] * S[i]
                dS_dt[i] += inflow[j, i] * S[j]
                dI_dt[i] -= outflow[i, j] * I[i]
                dI_dt[i] += inflow[j, i] * I[j]
                dR_dt[i] -= outflow[i, j] * R[i]
                dR_dt[i] += inflow[j, i] * R[j]

            dS_dt[i] -= self.beta[i] * S[i] * I[i] / self.N_i[temporal_verifier(self.day_elements, t, self.step), i]
            dI_dt[i] = self.beta[i] * S[i] * I[i] / self.N_i[temporal_verifier(self.day_elements, t, self.step), i] - self.gamma * I[i]
            dR_dt[i] = self.gamma * I[i]

        return np.hstack((dS_dt, dI_dt, dR_dt))

    def run_simulation(self):
        """
        Runs the simulation of the SIR model using the Eulerian approach.

        Returns
        -------
        numpy.ndarray
            A 2D array, where each row represents the values of the populations of each subpopulation at a specific time point.
        dict
            A dictionary containing information about the integration of the differential equations.
        """
        def save_integration_csv(self):
            """
            Saves the integration results as a CSV file.
            """
            S = self.Y[:, : self.K]
            I = self.Y[:, self.K : 2 * self.K]
            R = self.Y[:, 2 * self.K :]

            header = [f'{status} {i + 1}' for status in ('Susceptible', 'Infected', 'Removed') for i in range(self.K)]
            data = np.concatenate((S, I, R), axis=1)

            if not self.aggregated:
                dir = os.path.join(self.save_path, "SIRModel_integration_baseline")
                if not os.path.exists(dir):
                    os.makedirs(dir)
                file_path = os.path.join(dir, "SIRModel_integration_baseline.csv")
            else:
                dir = os.path.join(self.save_path, "SIRModel_integration_aggregated")
                if not os.path.exists(dir):
                    os.makedirs(dir)
                file_path = os.path.join(dir, f'SIRModel_integration_aggregated_{60 // self.aggregation_factor}.csv')

            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerows(data)

        self.Y = RK4(self._serial_system_ode, self.Y0, self.T, self.step)
        save_integration_csv(self)
    
    def R_0(self):
        """
        Calculates the basic reproduction number (R₀) for all the subpopulations.

        Returns
        -------
        R₀ : float
            The basic reproduction number.
        """

        return self.beta[0] / self.gamma
    
    def plot_system_ode(self):
        """
        Plots the results of the simulation for each subpopulation using the SIR model. The plot shows the susceptible,
        infected, and recovered populations over time for each subpopulation, with different colors for each population. The
        method saves the plot for each subpopulation in the directory specified by `save_path`. The beta and gamma
        values used for the simulation are also displayed on the plot for each subpopulation.
        """
        set_style('darkgrid')
        
        for i in range(self.K):
            fig, ax = plt.subplots(figsize=(15, 6))
            fig.suptitle(f'Subpopulation {i + 1} - Metapopulation SIR Model', fontsize=18)
            subtitle_text = r'$\beta$' + r'$\rightarrow$' + fr"{self.beta[i]}" + ' | ' + r'$\gamma$' + r'$\rightarrow$' + fr'{self.gamma}' + \
                            ' | ' + r'$R_{0}$' + r'$\rightarrow$' + fr'{self.basic_reproduction_number}'
            fig.text(0.5, 0.90, subtitle_text, ha='center', fontsize=14)
            
            ax.plot(self.T, self.Y[:, i], 'blue', lw=3, alpha=.7, label=r"Susceptible - $S(t)$")
            ax.plot(self.T, self.Y[:, self.K + i], 'red', lw=3, alpha=.7, label=r"Infected - $I(t)$")
            ax.plot(self.T, self.Y[:, 2 * self.K + i], 'green', lw=3, alpha=.7, label=r"Recovered - $R(t)$")
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Population Size', fontsize=12)
            ax.yaxis.set_tick_params(length=0)
            ax.xaxis.set_tick_params(length=0)
            ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
            legend = ax.legend()
            legend.get_frame().set_facecolor('white') 
            legend.get_frame().set_alpha(.7)
            for spine in ('top', 'right', 'bottom', 'left'):
                ax.spines[spine].set_visible(False)

            if not self.aggregated:
                dir = os.path.join(self.save_path, f"SIRModel_integration_baseline")
                fig.savefig(os.path.join(dir, f"SIRModel_integration_Subpopulation_{i + 1}_baseline.pdf"))
            else:
                dir = os.path.join(self.save_path, f"SIRModel_integration_aggregated")
                fig.savefig(os.path.join(dir, f"SIRModel_integration_Subpopulation_{i + 1}_aggregated_{60 // self.aggregation_factor}.pdf"))
            plt.close(fig)

def aggregate_networks(travel_rates, resolutions):
    """
    Aggregate network matrices according to different resolutions.

    Parameters
    ----------
    travel_rates : tuple(numpy.ndarray, numpy.ndarray)
        The tuple of inflow and outflow numpy.ndarrays that specifies the rates of movement between
        subpopulations. The first numpy.ndarray is the inflow rates, and the second numpy.ndarray is the outflow rates.
    resolutions : list of int
        List of resolutions for aggregation.

    Returns
    -------
    list of tuple(numpy.ndarray, numpy.ndarray)
        List of aggregated inflow and outflow matrices for each resolution.
    """
    def save_aggregated_networks(aggregated_networks):
        """
        Save aggregated network matrices as .npy files.

        Parameters
        ----------
        aggregated_networks : list of tuple(numpy.ndarray, numpy.ndarray)
            List of aggregated inflow and outflow matrices for different resolutions.
        save_path : str
            The path to save the aggregated network matrices.
        """
        if not os.path.exists('aggregated_networks'):
            os.makedirs('aggregated_networks')

        for i, (aggregated_inflows, aggregated_outflows) in enumerate(aggregated_networks):
            np.save(os.path.join('aggregated_networks', f'aggregated_inflows_res_{i + 1}.npy'), aggregated_inflows)
            np.save(os.path.join('aggregated_networks', f'aggregated_outflows_res_{i + 1}.npy'), aggregated_outflows)

    in_, out_ = travel_rates
    aggregated_networks = []

    for resolution in resolutions:
        aggregation_factor = 60 // resolution

        if resolution == 1:
            aggregated_inflows = np.mean(in_, axis=0)
            aggregated_outflows = np.mean(out_, axis=0)
        else:
            aggregated_inflows = []
            aggregated_outflows = []

            for i in range(0, len(in_), aggregation_factor):
                inflow_avg = np.mean(in_[i : i + aggregation_factor], axis=0)
                outflow_avg = np.mean(out_[i : i + aggregation_factor], axis=0)
                aggregated_inflows.append(inflow_avg)
                aggregated_outflows.append(outflow_avg)

        aggregated_networks.append((np.array(aggregated_inflows), np.array(aggregated_outflows)))
    
    save_aggregated_networks(aggregated_networks)
    return aggregated_networks

def generate_integration_scenarios(travel_rates, T, save_path, transition_rates):
    """
    Generate integration scenarios for the SIR model with different temporal resolutions.

    Parameters
    ----------
    travel_rates : tuple(numpy.ndarray, numpy.ndarray)
        The tuple of inflow and outflow numpy.ndarrays that specifies the rates of movement between
        subpopulations. The first numpy.ndarray is the inflow rates, and the second numpy.ndarray is the outflow rates.
        ...
    T : tuple(numpy.ndarray, float) 
        A tuple containing the time grid and step size. The first element is a 1D
        numpy array specifying the time points at which the simulation should be run. The second element is
        a float representing the step size between time points.
    save_path : str
        The path to save the simulation results.
    transition_rates : tuple(numpy.ndarray, float)
        A tuple containing the transition rates for the SIR model. The first element is a 1D numpy array
        specifying the beta (transmission rate) for each subpopulation. The second element is a float representing
        the gamma (recovery rate) for the infection.

    Returns
    -------
    dict
        A dictionary containing comparison results for different resolutions. The keys are the resolutions,
        and the values are normalized Frobenius norms comparing the trajectories with the baseline.
    """
    inflows, outflows = travel_rates
    beta, gamma = transition_rates
    time_grid, step = T
    resolutions = [i for i in range(60, 0, -1) if 60 % i == 0][1:]
    trajectories = {}

    baseline_model = SIRModel((inflows, outflows), (time_grid, step), save_path, beta, gamma, False, None)
    baseline_trajectory = baseline_model.Y
    trajectories['baseline'] = baseline_trajectory

    aggregated_networks = aggregate_networks(travel_rates, resolutions)

    for index, aggregated_network in enumerate(aggregated_networks):
        model = SIRModel(aggregated_network, (time_grid, step), save_path, beta, gamma, True, 60 // resolutions[index])
        trajectory = model.Y

        aggregated_aux = f'{int(60 // model.aggregation_factor)}' 
        trajectories['aggregated_' + aggregated_aux] = trajectory

    with open(os.getcwd() + '/' + save_path + '/trajectories.pkl', 'wb') as file:
        pk.dump(trajectories, file)

    return trajectories

def extract_subpopulation_norms(trajectories):
    """
    Extract subpopulation norms from a dictionary of integration results.

    Parameters:
    ----------
    trajectories : dict
        A dictionary containing integration results for different subpopulations.
        Each subpopulation is represented as a key in the dictionary, and the value
        associated with each key should be a NumPy ndarray of shape (6000, 909).

    Returns:
    -------
    dict
        A dictionary of subpopulation Frobenius norms where each subpopulation is represented
        as a key, and the corresponding value is a dict of Frobenius norms for each subtrajectory.
    """
    subtrajectories = {'baseline' : {}}
    dim_1, dim_2 = trajectories['baseline'].shape
    dim_3 = dim_2 // 303
    dim_4 = dim_2 // dim_3

    for i in range(dim_4):
        baseline_trajectory = trajectories['baseline']
        baseline_subtrajectory = np.empty((dim_1, dim_3))
        baseline_subtrajectory[:, 0] = baseline_trajectory[:, i]
        baseline_subtrajectory[:, 1] = baseline_trajectory[:, dim_4 + i]
        baseline_subtrajectory[:, 2] = baseline_trajectory[:, (dim_4 * 2) + i]
        subtrajectories['baseline'][i] = frobenius_norm(baseline_subtrajectory)

    for i in range(dim_4):
        for key, subtrajectory in trajectories.items():
            if key != 'baseline':
                if key not in subtrajectories:
                    subtrajectories[key] = {}
                aggregated_subtrajectory = np.empty((dim_1, dim_3))
                aggregated_subtrajectory[:, 0] = subtrajectory[:, i]
                aggregated_subtrajectory[:, 1] = subtrajectory[:, dim_4 + i]
                aggregated_subtrajectory[:, 2] = subtrajectory[:, (dim_4 * 2) + i]
                subtrajectories[key][i] = frobenius_norm(aggregated_subtrajectory)
    
    with open('subtrajectories_norms.pkl', 'wb') as file:
        pk.dump(subtrajectories, file)

    return subtrajectories

def find_city_names(nodes_codes_file, nodes_cities_file):
    """
    Find and return a list of city names based on node codes.

    Parameters:
    nodes_codes_file (str): The path to a CSV file containing node codes.
    nodes_cities_file (str): The path to a CSV file containing city information.

    Returns:
    list: A list of city names corresponding to the node codes found in nodes_codes_file.
            If a code is not found, 'Not Found' is used as a placeholder.
    """
    nodes_codes = np.loadtxt(nodes_codes_file, delimiter=';')[:, 0]
    nodes_codes = [str(int(code)) if code.is_integer() else str(code) for code in nodes_codes]
    nodes_cities = np.loadtxt(nodes_cities_file, delimiter=',', dtype='str')[1:, :]
    city_names = []

    for code in nodes_codes:
        matching_indices = np.where(nodes_cities[:, 0] == code)[0]
        city_name = nodes_cities[matching_indices, 3][0]
        city_names.append(city_name)

    return city_names

def calculate_node_centralities(inflow_matrix, outflow_matrix, city_names):
    """
    Calculate node centralities for a single day's inflow and outflow matrices.

    Parameters:
    ----------
    inflow_matrix : numpy.ndarray
        A 2D NumPy array representing the inflow matrix for a single day.
    outflow_matrix : numpy.ndarray
        A 2D NumPy array representing the outflow matrix for a single day.

    Returns:
    -------
    dict
        A dictionary containing the node centralities for inflow and outflow for that day.
    """
    inflow_graph = nx.DiGraph(inflow_matrix)
    outflow_graph = nx.DiGraph(outflow_matrix)

    inflow_centralities = nx.degree_centrality(inflow_graph)
    outflow_centralities = nx.degree_centrality(outflow_graph)

    inflow_centralities_by_city = {city_names[i]: centrality for i, centrality in enumerate(inflow_centralities)}
    outflow_centralities_by_city = {city_names[i]: centrality for i, centrality in enumerate(outflow_centralities)}

    return {
        'inflow': inflow_centralities_by_city,
        'outflow': outflow_centralities_by_city
    }

def calculate_centralities(travel_rates):
    """
    Calculate node centralities for inflow and outflow matrices day by day in parallel.

    Parameters:
    ----------
    travel_rates : tuple of numpy.ndarray
        A tuple containing two 3D NumPy arrays representing inflow and outflow matrices with dimensions (T, N, N),
        where T is the number of days, and N is the number of nodes.

    Returns:
    -------
    dict
        A dictionary where keys are days (from 0 to T-1) and values are dictionaries containing the node centralities
        for inflow and outflow for that day.
    """
    inflow, outflow = travel_rates
    T, N, _ = inflow.shape
    city_names = find_city_names('degree.csv', 'Index_City_CH_EN.csv')

    pool = mp.Pool(processes=mp.cpu_count())

    # Calculate node centralities in parallel for each day
    centralities = {}
    results = []

    for t in range(T):
        inflow_matrix = inflow[t]
        outflow_matrix = outflow[t]
        results.append(pool.apply_async(calculate_node_centralities, args=(inflow_matrix, outflow_matrix, city_names)))

    for t, result in enumerate(results):
        centralities[t] = result.get()

    pool.close()
    pool.join()

    return centralities