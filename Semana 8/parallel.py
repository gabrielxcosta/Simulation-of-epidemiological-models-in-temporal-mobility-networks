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
        with mp.Pool(mp.cpu_count() - 1) as pool:
            results = pool.map(self._ode_worker, [(i, N, outflow, inflow) for i in range(self.K)])
        dN_dt = np.array(results)
        return dN_dt

    def _system_ode_worker(self, args):
        """
        Worker function to compute the derivative of each compartment in a single system of ODEs.

        Args:
            args (tuple): Tuple containing the arguments.
                i (int): Index of the compartment.
                y (numpy.ndarray): Array of compartment values.

        Returns:
            tuple: Tuple containing the computed derivatives for each compartment.
                dS_dt (float): Derivative of the susceptible compartment.
                dI_dt (float): Derivative of the infected compartment.
                dR_dt (float): Derivative of the recovered compartment.
        """
        i, y, outflow, inflow = args
        S = y[:self.K]
        I = y[self.K : 2 * self.K]
        R = y[2 * self.K:]

        dS_dt = 0.0
        dI_dt = 0.0
        dR_dt = 0.0

        for j in range(self.K):
            dS_dt -= outflow[i, j] * S[i]
            dS_dt += inflow[j, i] * S[j]
            dI_dt -= outflow[i, j] * I[i]
            dI_dt += inflow[j, i] * I[j]
            dR_dt -= outflow[i, j] * R[i]
            dR_dt += inflow[j, i] * R[j]

        dS_dt -= self.beta[i] * S[i] * I[i] / self.N_i[-1, i]
        dI_dt = self.beta[i] * S[i] * I[i] / self.N_i[-1, i] - self.gamma * I[i]
        dR_dt = self.gamma * I[i]

        return dS_dt, dI_dt, dR_dt

    def _parallel_system_ode(self, y, t):
        """
        Compute the derivatives of each compartment in parallel for a system of ODEs.

        Args:
            y (numpy.ndarray): Array of compartment values.
            t (float): Current time point.

        Returns:
            numpy.ndarray: Array containing the derivatives of each compartment.
        """
        print(f"Dia {temporal_verifier(self.day_elements, t, self.step) + 1}:")
        outflow = self.outflow_rates[temporal_verifier(self.day_elements, t, self.step)]
        inflow = self.inflow_rates[temporal_verifier(self.day_elements, t, self.step)]

        dS_dt = np.zeros(self.K)
        dI_dt = np.zeros(self.K)
        dR_dt = np.zeros(self.K)

        with mp.Pool(mp.cpu_count() - 1) as pool:
            results = pool.map(self._system_ode_worker, ((i, y, outflow, inflow) for i in range(self.K)))

        for i, result in enumerate(results):
            dS_dt[i], dI_dt[i], dR_dt[i] = result

        return np.hstack((dS_dt, dI_dt, dR_dt))