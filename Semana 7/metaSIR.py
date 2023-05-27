import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from seaborn import set_palette, set_style
from scipy.integrate import odeint

def generate_travel_rates(N, save_path):
    """
    Generates an NxN matrix of travel rates between N locations, where each row 
    represents the travel rates from one location to all other locations.

    Parameters
    ----------
    N : int
        Number of locations.
    save_path : str
        The path to the directory where the generated CSV file will be saved.

    Returns
    -------
    f_matrix : numpy.ndarray
        An NxN matrix of travel rates, where f_matrix[i,j] represents the travel 
        rate from location i to location j.
    """

    # Create an empty NxN matrix
    f_matrix = np.zeros((N, N))
    for i in range(N):
        # Generate N random numbers between 0 and 1
        row = np.random.rand(N)
        # Set the ith element of the row to 0
        row[i] = 0.0
        # Ensure that all elements are positive
        row = np.abs(row)
        # Normalize the row to ensure that the sum is equal to 1
        row = row / np.sum(row)
        # Set the row in the matrix
        f_matrix[i, :] = row

    # If the folder does not exist, we create it
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Save the numpy array to a CSV file
    np.savetxt(save_path + '/travel_rates.csv', f_matrix, delimiter=',')

    return f_matrix

def create_network_from_OD(OD_matrix):
    """
    Creates a weighted directed graph (digraph) using NetworkX from an origin-destination matrix.

    Parameters:
    OD_matrix (numpy.ndarray): The origin-destination matrix representing the weights of edges between nodes.

    Returns:
    nx.DiGraph: The weighted directed graph (digraph) created from the origin-destination matrix.
    """
    G = nx.DiGraph()
    num_nodes = OD_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = OD_matrix[i, j]
            if weight != 0:
                G.add_edge(i, j, weight=weight)

    return G

def plot_network(G, save_path):
    """
    Plots the given network graph using NetworkX and saves the plot as an image.

    Parameters:
    G (nx.Graph or nx.DiGraph): The network graph to be plotted.
    save_path (str): The path to save the plot image.

    Returns:
    None
    """
    plt.figure(figsize=(12, 6), dpi=300)
    plt.suptitle('Metapopulation Commuting Network', fontsize=20, fontdict={'color': 'white'})
    plt.title(fr'$K = {len(G.nodes())}$', fontsize=14, fontdict={'color': 'white'})

    with plt.style.context('dark_background'):

        pos = nx.spring_layout(G)  # Compute the layout once

        nx.draw(
            G,
            pos=pos,
            width=2,
            node_color='white',
            font_color='black',
            alpha=1
        )

        labels = {n: n + 1 for n in G.nodes()}
        nx.draw_networkx_labels(
            G,
            pos=pos,
            labels=labels,
            font_size=12,
            font_color='black'
        )

        edges = G.edges()

        nx.draw_networkx_edges(
            G,
            pos=pos,
            edgelist=edges,
            edge_color='white',
            alpha=.5,
            style='solid'
        )
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        return plt.savefig(save_path + '/net.png')

class FluxModel:
    """
    A class for simulating the movement of individuals between subpopulations over time, using the Eulerian
    approach. The model is based on a flux matrix that describes the rates of movement between subpopulations,
    and assumes that the population sizes within each subpopulation are continuous and homogeneous.

    Parameters
    ----------
    travel_rates : numpy.ndarray
        An NxN matrix where N is the number of subpopulations that specifies the rates of movement between
        subpopulations. The (i,j)-th element of the matrix denotes the rate of movement from subpopulation i
        to subpopulation j. The diagonal of the matrix should be equal to 0.0.
    t : numpy.ndarray
        A 1D array specifying the time points at which the simulation should be run.
    save_path : str
        The path to save the simulation results.

    Attributes
    ----------
    travel_rates : numpy.ndarray
        The flux matrix that specifies the rates of movement between subpopulations.
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
    """
    def __init__(self, travel_rates, t, save_path):
        """
        Initializes a new instance of the FluxModel class with the given parameters, and runs the simulation.

        Args:
            travel_rates (numpy.ndarray): An NxN matrix where N is the number of subpopulations that specifies
                the rates of movement between subpopulations.
            t (numpy.ndarray): A 1D array specifying the time points at which the simulation should be run.
            save_path (str): The path to save the simulation results.
        """
        self.travel_rates = travel_rates
        self.t = t
        self.save_path = save_path
        self.K = len(travel_rates)
        self.N0 = np.ones(self.K) * 500
        self.N = np.sum(self.N0)
        self.N_i, self.output = None, None

        self.G = create_network_from_OD(self.travel_rates)
        plot_network(self.G, save_path)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.simulate() # Getting the data of self.N_i and self.output
        self.plot_diffusion()

    def _ode(self, N, t):
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
        dN_dt = np.zeros(self.K)
        for i in range(self.K):
            outgoing_flux = 0.0
            incoming_flux = 0.0
            for j in range(self.K):
                outgoing_flux += self.travel_rates[i, j] * N[i]
                incoming_flux += self.travel_rates[j, i] * N[j]
            dN_dt[i] = incoming_flux - outgoing_flux
        return dN_dt

    def simulate(self):
        '''
        Simulates the movement of individuals between subpopulations over time, using the Eulerian approach. 
        Returns a 2D array of the population sizes of each subpopulation at each time point.

        Parameters
        ----------
        t : numpy.ndarray
            A 1D array that specifies the time points at which the population sizes are computed.

        save_path : str
            The directory path to save the integration.csv file.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (len(t), K) that specifies the population sizes of each subpopulation at each 
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

        self.N_i, self.output = odeint(self._ode, self.N0, self.t, full_output=1)
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
        fig = plt.figure(figsize=(20, 8), facecolor='w', dpi=300)
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

        for i in range(self.K):
            ax.plot(self.t, self.N_i[:, i], label=f'Subpopulation {i + 1}', lw=4, alpha=.8)

        ax.set_title('Flux model - Eulerian movement model', fontsize=22)
        ax.set_xlabel('Time', fontsize=18)
        ax.set_ylabel('Population Size', fontsize=18)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(.7)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)

        return fig.savefig(os.getcwd() + '/' + self.save_path + '/flux_model_results.png')

class SIRModel(FluxModel):
    """
    A class for simulating the Susceptible-Infected-Recovered (SIR) model with host diffusion among metapopulations,
    using the Eulerian approach. The model is based on a flux matrix that describes the rates of movement between
    subpopulations, and assumes that the population sizes within each subpopulation are continuous and homogeneous.

    Parameters
    ----------
    travel_rates : numpy.ndarray
        An NxN matrix where N is the number of subpopulations that specifies the rates of movement between
        subpopulations. The (i,j)-th element of the matrix denotes the rate of movement from subpopulation i
        to subpopulation j. The diagonal of the matrix should be equal to 0.0.
    t : float
        The final time point of the simulation.
    save_path : str
        The path to save the simulation results.
    beta : numpy.ndarray
        A 1D array with length N that specifies the transmission rate for each subpopulation.
    gamma : float
        The recovery rate for the infection.

    Attributes
    ----------
    beta : numpy.ndarray
        The transmission rate for each subpopulation.
    gamma : float
        The recovery rate for the infection.
    travel_rates : numpy.ndarray
        The flux matrix that specifies the rates of movement between subpopulations.
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
    output : dict or None
        A dictionary containing information about the integration of the differential equations. It is set
        to None initially.
    """
    def __init__(self, travel_rates, t, save_path, beta, gamma):
        """
        Initializes a new instance of the SIRModel class with the given parameters, and runs the simulation.

        Args:
            travel_rates (numpy.ndarray): An NxN matrix where N is the number of subpopulations that specifies the rates
                of movement between subpopulations.
            t (float): The final time point of the simulation.
            save_path (str): The path to save the simulation results.
            beta (numpy.ndarray): A 1D array with length N that specifies the transmission rate for each subpopulation.
            gamma (float): The recovery rate for the infection.
        """
        super().__init__(travel_rates, t, save_path)
        self.beta = np.asarray(beta)
        self.gamma = gamma
        self.Y0 = np.hstack((self.N0, self.N0 - 400, np.zeros(self.K)))
        self.Y = None
        self.basic_reproduction_number = self.R_0()
        self.output = None
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.run_simulation() # Getting the data of self.Y and self.output
        self.plot_results()
    
    def R_0(self):
        """
        Calculates the basic reproduction number (R₀) for a disease.

        Parameters
        ----------
        beta : np.array
            A NumPy array of the transmission rates for each age group.
        gamma : float
            The recovery rate.

        Returns
        -------
        R₀ : float
            The basic reproduction number.

        """

        return self.beta / self.gamma

    def _system_ode(self, y, t):
        """
        Computes the system of ordinary differential equations for the SIR model.

        Parameters
        ----------
        y : numpy.ndarray
            A 1D array with length 3K that specifies the initial values of the populations for each subpopulation.
            The first K elements of the array correspond to the initial values of the susceptible population for each subpopulation,
            the next K elements correspond to the initial values of the infected population for each subpopulation,
            and the last K elements correspond to the initial values of the recovered population for each subpopulation.
        t : numpy.ndarray
            A 1D array specifying the time points at which the simulation should be run.

        Returns
        -------
        numpy.ndarray
            A 1D array that specifies the values of the derivatives of the populations at each time point.
        """
        S = y[:self.K]
        I = y[self.K : 2 * self.K]
        R = y[2 * self.K:]

        dSdt = [-self.beta[i] * S[i] * I[i] / self.N_i[-1, i] - \
                np.sum([self.travel_rates[i, j] * S[i] for j in range(self.K)]) + \
                np.sum([self.travel_rates[j, i] * S[j] for j in range(self.K)]) for i in range(self.K)
            ]

        dIdt = [self.beta[i] * S[i] * I[i] / self.N_i[-1, i] - self.gamma * I[i] - \
                np.sum([self.travel_rates[i, j] * I[i] for j in range(self.K)]) + 
                np.sum([self.travel_rates[j, i] * I[j] for j in range(self.K)]) for i in range(self.K)
            ]

        dRdt = [self.gamma * I[i] - \
                np.sum([self.travel_rates[i, j] * R[i] for j in range(self.K)]) +  
                np.sum([self.travel_rates[j, i] * R[j] for j in range(self.K)]) for i in range(self.K)
            ]

        return np.hstack((dSdt, dIdt, dRdt))

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
            
            with open(self.save_path + '/SIRModel_integration.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerows(data)

        self.Y, self.output = odeint(self._system_ode, self.Y0, self.t, full_output=1)
        save_integration_csv(self)

    '''
    def plot_results(self):
        """
        Plots the results of the simulation for each subpopulation using the SIR model. The plot shows the susceptible,
        infected, and recovered populations over time for each subpopulation, with different colors for each population. The
        method saves the plot for each subpopulation in the directory specified by `save_path`. The beta and gamma
        values used for the simulation are also displayed on the plot for each subpopulation.
        """   
        for i in range(self.K):
            fig = plt.figure(figsize=(20, 8), facecolor='w', dpi=300)
            ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
            ax.plot(self.t, self.Y[:, i], 'blue', lw=3, alpha=.7, label="Susceptible")
            ax.plot(self.t, self.Y[:, self.K + i], 'red', lw=3, alpha=.7, label="Infected")
            ax.plot(self.t, self.Y[:, 2 * self.K + i], 'green', lw=3, alpha=.7, label="Recovered")
            ax.figure.suptitle(f"SIR Model - Subpopulation {i + 1}", fontsize=22)
            ax.set_title(r'$\beta$' + r'$\rightarrow$' + fr"{self.beta[i]}" + ' | ' + r'$\gamma$' + r'$\rightarrow$' + fr'{self.gamma}', fontsize=18)
            ax.set_xlabel('Time', fontsize=18)
            ax.set_ylabel('Population Size', fontsize=18)
            ax.yaxis.set_tick_params(length=0)
            ax.xaxis.set_tick_params(length=0)
            ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
            legend = ax.legend()
            legend.get_frame().set_alpha(.7)
            for spine in ('top', 'right', 'bottom', 'left'):
                ax.spines[spine].set_visible(False)
            fig.savefig(os.path.join(self.save_path, f"Subpopulation_{i + 1}.png"))
            plt.close(fig)
    '''

    def plot_results(self):
        """
        Plots the results of the simulation for each subpopulation using the SIR model. The plot shows the susceptible,
        infected, and recovered populations over time for each subpopulation, with different colors for each population. The
        method saves the plot for each subpopulation in the directory specified by `save_path`. The beta and gamma
        values used for the simulation are also displayed on the plot for each subpopulation.
        """
        set_style('darkgrid')
        fig, axes = plt.subplots(self.K, 1, figsize=(15, 20), dpi=300)
        fig.suptitle('Metapopulation SIR Model in a Temporal Network', fontsize=22)
        subtitle_text = r'$\beta$' + r'$\rightarrow$' + fr"{self.beta}" + ' | ' + r'$\gamma$' + r'$\rightarrow$' + fr'{self.gamma}' + \
                        ' | ' + r'$R_{0}$' + r'$\rightarrow$' + fr'{self.basic_reproduction_number}'
        fig.text(0.5, 0.95, subtitle_text, ha='center', fontsize=18)  # Add the subtitle below the suptitle
        fig.subplots_adjust(hspace=0.2)
        
        for i, ax in enumerate(axes):
            ax.plot(self.t, self.Y[:, i], 'blue', lw=3, alpha=.7, label=r"Susceptible - $S(t)$")
            ax.plot(self.t, self.Y[:, self.K + i], 'red', lw=3, alpha=.7, label=r"Infected - $I(t)$")
            ax.plot(self.t, self.Y[:, 2 * self.K + i], 'green', lw=3, alpha=.7, label=r"Recovered - $R(t)$")
            ax.set_title(f"Subpopulation {i + 1}", fontsize=18)
            ax.set_xlabel('Time', fontsize=14)
            ax.set_ylabel('Population Size', fontsize=14)
            ax.yaxis.set_tick_params(length=0)
            ax.xaxis.set_tick_params(length=0)
            ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
            legend = ax.legend()
            legend.get_frame().set_facecolor('white') 
            legend.get_frame().set_alpha(.7)
            for spine in ('top', 'right', 'bottom', 'left'):
                ax.spines[spine].set_visible(False)

        fig.savefig(os.path.join(self.save_path, "Subpopulations.png"))
        plt.close(fig)