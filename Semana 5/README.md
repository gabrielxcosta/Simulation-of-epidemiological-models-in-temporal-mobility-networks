

## Epidemic dynamics on metapopulation networks
 
Metapopulation models are a powerful tool for theorizing and simulating
epidemic dynamics. One possibility of implementation is to have nodes as sub-
populations of a given location and the links representing migrations of individ-
uals, obeying a specific mobility rule, from one subpopulation to another in a
given time window.

We simulate a SIR metapopulation model on top of a temporal Chinese
commuting network, where nodes are cities and weighted connections give the
rate of people that reaches or leaves a city. Each node has its metapopulation
that interacts with its neighbors. Hence, people may infect and get infected
depending on where they meet: i) infection of susceptibles in a certain city by
infected people who live there; ii) infected people from other nodes who arrive
at the destination and infect the locals; iii) susceptibles who traveled to another
node and get infected by the locals; iv) susceptible people that go to a place
and meet infected that comes from a third site.

The commuting network has a temporal resolution of 24 hours, which means
the network updates its connections once a day. We analyze the dependence
of epidemiological variables on the network resolution, by simulating the SIR
model under different temporally aggregated versions of the network.

The simulations may help to design strategies for commuting data collection
concerning temporal resolution and its impacts on epidemiological surveillance.

<p align="center">
 <img src='MetaSIR2.jpeg'>
</p>
