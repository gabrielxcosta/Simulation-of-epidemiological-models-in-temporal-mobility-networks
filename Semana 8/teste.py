import os
import re
import csv
import numpy as np
import networkx as nx
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from seaborn import set_palette, set_style

def integrate_rk4(func, y0, T, step):
    """
    Fourth Order Runge-Kutta method for integrating a system of ordinary differential equations.

    Args:
        func (callable): Function that computes the time derivative of the state variables.
        y0 (array-like): Initial conditions of the system.
        T (array-like): Time grid.
        step (float): Step size for integration.

    Returns:
        y (array-like): Array containing the state variables at each time point.
    """
    y = np.zeros((len(T), len(y0)))
    y[0] = y0
    
    dt = step
    print('dt = ', dt)

    for i in range(len(T) - 1):
        t = T[i]
        k1 = func(y[i], t) 
        k2 = func(y[i] + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = func(y[i] + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = func(y[i] + dt * k3, T[i + 1])
        y[i + 1] = y[i] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        print(T[i], y[i + 1] - y[i])

    return y

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
        np.savetxt(os.path.join(save_path, f'inflow_travel_rates_day{day + 1}.csv'), inflow_matrix, delimiter=',')
        np.savetxt(os.path.join(save_path, f'outflow_travel_rates_day{day + 1}.csv'), outflow_matrix, delimiter=',')

    return inflow_matrices, outflow_matrices

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

def get_OD_matrices(folder_name):
    '''
    Process inflow and outflow graphml files in parallel and return their adjacency matrices.

    Parameters:
        folder_name (str): The path to the folder containing the graphml files.

    Returns:
        tuple: A tuple containing two numpy arrays representing inflow and outflow adjacency matrices.
            Tuple -> (inflow, outflow)
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

def identify_day_beginnings(time_grid, dt):
    """
    Identifies the indices in the time grid that represent the beginning of each day.

    Parameters:
        time_grid (numpy.ndarray): The array of time grid.

    Returns:
        numpy.ndarray: An array of indices representing the beginning of each day.

    """
    return np.array([int(i * (len(time_grid) / 60)) for i in range(60)])

def capture_elements_per_day(time_grid, day_beginnings):
    """
    Captures the elements in the interval of each day based on the day beginnings.

    Parameters:
        time_grid (numpy.ndarray): The array of time grid.
        day_beginnings (numpy.ndarray): An array of indices representing the beginning of each day.

    Returns:
        numpy.ndarray: An array of elements in the interval of each day.

    """
    day_elements = [time_grid[start:end] for start, end in zip(day_beginnings, day_beginnings[1:])]
    day_elements.append(time_grid[day_beginnings[-1]:])
    return np.array(day_elements, dtype=object)

def temporal_verifier(arrays, value, dt):
    """
    Checks the index of the array (day) containing the given value (flow).

    Parameters:
        arrays (numpy.ndarray): Array of arrays.
        value: The value to search for.

    Returns:
        int: Index of the array containing the value, or -1 if not found.

    """
    arrays_flat = np.concatenate(arrays)
    indices = np.where(arrays_flat == value)[0]
    # in case the timestep is now found, it means it is probably an intermediate value, due to RK4 midpoints computations.
    if(indices.size == 0):
        # in this case, one should use t - dt/2 to capture the previous time step.
        indices = np.where(arrays_flat == value - dt / 2.0)[0]
    
    index = indices[0]
    array_index = np.searchsorted(np.cumsum([len(arr) for arr in arrays]), index, side='right')
    return array_index

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
    # in case the timestep is now found, it means it is probably an intermediate value, due to RK4 midpoints computations.
    if indices.size == 0:
        # in this case, one should use t - dt/2 to capture the previous time step.
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
        The tuple of inflow and outflow numpy.ndarrays that specifies the rates of movement between
        subpopulations. The first numpy.ndarray is the inflow rates, and the second numpy.ndarray is the outflow rates.
        The (i,j)-th element of the inflow rates numpy.ndarray denotes the rate of movement from subpopulation i
        to subpopulation j. The (i,j)-th element of the outflow rates numpy.ndarray denotes the rate of movement from subpopulation j
        to subpopulation i. The diagonal of both numpy.ndarrays should be equal to 0.0.
    T : numpy.ndarray
        A 1D array specifying the time points at which the simulation should be run.
    save_path : str
        The path to save the simulation results.

    Attributes
    ----------
    travel_rates : tuple(numpy.ndarray, numpy.ndarray)
        The tuple of inflow and outflow numpy.ndarrays that specifies the rates of movement between subpopulations
        for each day of the temporal network.
    G : nx.DiGraph
        The weighted directed graph (digraph) created from the origin-destination matrix.
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
    output : dict or None
        A dictionary containing information about the integration of the differential equation. It is set
        to None initially.

    Methods
    -------
    __init__(self, travel_rates, T, save_path)
        Initializes a new instance of the FluxModel class with the given parameters, and runs the simulation.
    simulate(self)
        Runs the simulation and returns the population sizes at each time step.
    plot_diffusion(self)
        Plots the diffusion of individuals between subpopulations over time.
    """
    def __init__(self, travel_rates, T, save_path):
        """
        Initializes a new instance of the FluxModel class with the given parameters, and runs the simulation.

        Args:
            travel_rates (numpy.ndarray): An NxN matrix where N is the number of subpopulations that specifies
                the rates of movement between subpopulations.
            T (numpy.ndarray): A 1D array specifying the time points at which the simulation should be run.
            save_path (str): The path to save the simulation results.
        """
        self.inflow_rates, self.outflow_rates = travel_rates
        self.T, self.step = T
        self.save_path = save_path
        self.K = len(self.inflow_rates[0])
        self.N0 = np.ones(self.K) * 500
        self.N = np.sum(self.N0)
        self.N_i, self.output = None, None
        self.day_beginnings = identify_day_beginnings(self.T, self.step)
        self.day_elements = capture_elements_per_day(self.T, self.day_beginnings)

        #plot_network(self.G, save_path)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.simulate() # Getting the data of self.N_i and self.output
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
        outflow = self.outflow_rates[temporal_verifier(self.day_elements, t, self.step)]
        inflow = self.inflow_rates[temporal_verifier(self.day_elements, t, self.step)]

        dN_dt = np.zeros(self.K)
        for i in range(self.K):
            outgoing_flux = 0.0
            incoming_flux = 0.0
            for j in range(self.K):
                outgoing_flux += outflow[i, j] * N[i]
                incoming_flux += inflow[j, i] * N[j]
                print('ANTES:', incoming_flux - outgoing_flux)
            dN_dt[i] = incoming_flux - outgoing_flux
            print('DEPOIS:', dN_dt[i])
        return dN_dt
    
    def _ode_worker(self, args):
        """
        Worker function for parallel computation of the time derivative of the number of hosts in each metapopulation.
        Computes the flux of hosts based on the outflow and inflow rates.

        Args:
            args: Tuple containing the index i and the current number of hosts N.

        Returns:
            The time derivative of the number of hosts in the i-th metapopulation.
        """
        i, N, outflow, inflow = args
        print(inflow)
        print()
        print(outflow)
        outgoing_flux = 0.0
        incoming_flux = 0.0
        for j in range(self.K):
            outgoing_flux += outflow[i, j] * N[i]
            incoming_flux += inflow[j, i] * N[j]
        return incoming_flux - outgoing_flux

    def _parallel_ode(self, N, t):
        """
        Computes the time derivative of the number of hosts in each metapopulation in parallel.
        Uses multiprocessing to distribute the computation across multiple processes.

        Args:
            N (array-like): Array containing the current number of hosts in each metapopulation.
            t (float): Current time point.

        Returns:
            dN_dt (array-like): Array containing the time derivative of the number of hosts in each metapopulation.
        """
        outflow = self.outflow_rates[temporal_verifier(self.day_elements, t, self.step)]
        inflow = self.inflow_rates[temporal_verifier(self.day_elements, t, self.step)]

        dN_dt = np.zeros(self.K)
        with mp.Pool() as pool:
            results = pool.map(self._ode_worker, [(i, N, outflow, inflow) for i in range(self.K)])
        dN_dt = np.array(results)
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
            # Write data to CSV file
            with open(self.save_path + '/FluxModel_integration.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                header = ['Subpopulation {}'.format(i + 1) for i in range(self.K)]
                writer.writerow(header)
                for i in range(len(self.N_i)):
                    writer.writerow(self.N_i[i])

        self.N_i = integrate_rk4(self._serial_ode, self.N0, self.T, self.step)
        save_integration_csv(self)
    
    def plot_diffusion(self):
        '''
        Plots the population sizes of each subpopulation over time, and saves the figure to a file.
        
        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object that contains the plot.
        '''
        set_palette('husl')
        fig = plt.figure(figsize=(20, 8), facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

        for i in range(self.K):
            ax.plot(self.T, self.N_i[:, i], label=f'Subpopulation {i + 1}', lw=4)

        ax.set_title('Diffusion of hosts among subpopulations', fontsize=22)
        ax.set_xlabel('Time', fontsize=18)
        ax.set_ylabel(r'$\frac{dN_{i}}{dt}$', fontsize=18)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(.7)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)

        return fig.savefig(os.getcwd() + '/' + self.save_path + '/flux_model_results.pdf')
    
inflow, outflow = generate_travel_rates(3, 'synthetic_data', 60)
#T, s = np.linspace(0, 60 ** 3 * 24, 1000, retstep=True)

s = 0.15
#T = np.arange(0, 60 ** 3 * 24, s)
T = np.arange(0, 60 ** 2 * 24 * 30, s)

print("Time instants: ", T)
print("s = ", s)

db = identify_day_beginnings(T, s)
print('db = ', db)
#print()
#print(capture_elements_per_day(T, db))

FluxModel(
    (inflow, outflow), 
    (T, s),
    'FluxModel')