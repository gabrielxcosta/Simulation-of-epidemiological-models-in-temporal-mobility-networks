import math as m
import numpy as np
from scipy.integrate import solve_ivp

class FluxModel:
    def __init__(self, N, inflow_matrices, outflow_matrices):
        """
        Inicializa o modelo de fluxo.

        Parâmetros:
        N : array_like
            Vetor inicial de populações em cada nó.
        inflow_matrices : list of numpy.ndarray
            Lista de matrizes NxN das taxas de entrada para cada dia.
        outflow_matrices : list of numpy.ndarray
            Lista de matrizes NxN das taxas de saída para cada dia.
        """
        self.N = N
        self.inflow_matrices = inflow_matrices
        self.outflow_matrices = outflow_matrices
        self.days = len(inflow_matrices)
        self.last_day = float(self.days)

    def omega(self, t, N):
        """
        Calcula a taxa de mudança de hospedeiros nas subpopulações baseada nas matrizes de fluxo,
        operador de transporte Ω

        Parâmetros:
        t : float
            Tempo atual (em dias).
        N : array_like
            Vetor atual de populações em cada nó.

        Retorna:
        dNdt : array_like
            Taxa de mudança de N.
        """
        day = m.floor(t)  # Ajuste para arredondar o tempo para o dia mais próximo
        if t == self.last_day:
            day = int(self.last_day - 1.)
        print('CURRENT TIME POINT: ', t, '- CALCULATED DAY: ', day)  # Debug para verificar o tempo e o dia calculado
        inflow_matrix = self.inflow_matrices[day]
        outflow_matrix = self.outflow_matrices[day]

        dN_dt = np.zeros_like(N)

        for i in range(len(N)):
            incoming = np.sum(inflow_matrix[i, :] * N)
            outgoing = np.sum(outflow_matrix[:, i] * N)
            dN_dt[i] = incoming - outgoing

        return dN_dt

    def solve(self, t_span, N0, num_points=25000, method='RK45'):
        """
        Resolve o sistema de equações diferenciais com um número fixo de pontos de integração.

        Parâmetros:
        t_span : tuple
            Intervalo de tempo para a integração (t0, tf).
        N0 : array_like
            Vetor inicial de populações em cada nó.
        num_points : int, optional
            Número de pontos de integração ao longo do intervalo de tempo (padrão é 25000).
        method : str, optional
            Método de integração a ser usado por solve_ivp (padrão é 'RK45').

        Retorna:
        result : OdeResult
            Objeto com a solução.
        """
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        sol = solve_ivp(self.omega, t_span, N0, method=method, t_eval=t_eval, dense_output=True)
        return sol

    def check_constraint(self, solution):
        if np.allclose(np.sum(solution.y, axis=0), np.sum(self.N)):
            print('Constraint satisfied!')
        else:
            print('Constraint not satisfied!')
