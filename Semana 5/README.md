

## Epidemic dynamics on metapopulation networks
 
Metapopulation models have been a powerful tool for both theorizing and simulating ``epidemic dynamics``. In
a metapopulation model, one considers a network composed of subpopulations and their pairwise connections,
and individuals are assumed to migrate from one subpopulation to another obeying a given mobility rule. While
how different mobility rules affect epidemic dynamics in metapopulation models has been studied, there have been
relatively few efforts on comparison of the effects of simple (i.e., unbiased) random walks and more complex
mobility rules. 

A metapopulation model assumes that a population of individuals is distributed over subpopulations, which correspond to the geographical locations, such as local gathering places, cities, or counties. On the microscopic scale, the individuals are assumed to be fully mixed within each subpopulation. An infectious individual infects each susceptible individual in the same subpopulation with the same rate/probability. This assumption is practical in the absence of detailed data on the structure of interactions among the individuals within each subpopulation. On the macroscopic scale, the individuals traverse edges in the network to travel from one subpopulation to another according to a mobility rule.

In order to understand the geographical diffusion of diseases, one has to combine the microscopic contagion processes with the long-range disease propagation due to human mobility across different spatial scales. In order to tackle this problem, epidemic modeling has relied on reaction-diffusion dynamics in ``metapopulations``. Metapopulations can be thought as nodes of a complex network of spatial patches, where links encode human flows from one place to another and are responsible for between-patch transmission.

## Metapopulation SIR model - Considering flows of people between populations

<center>

![MetapopSIR](MetaSIR2.jpeg)

</center>

Modeling:

- $M$: number of patches;
- $N$: total number of agents;
- $N_i$: population of the i-th patch;
- At any time $N = \sum_{i} N_i$ (considering the system is closed, without births or deaths);

Let $(m_{ij}) \in \mathbb{R}^{N \times N}$ be the adjacency matrix of a mobility network with $N$ nodes, whose entries correspond to the flow of people from $i$ to $j$. Each node is a patch, with fixed coordinates and a metapopulation. This means each node has its own SIR compartments and their evolution depend not only on the inner population but also on other nodes.

The probability $P_{ij}$ that an agent placed in $i$ moves to $j$ must be proportional to the flux $m_{ij}$ and reads

$$P_{ij} = \frac{m_{ij}}{\sum_{j=1}^{M} m_{ij}}$$

## Considering flows of people between populations

|        | $I^{O}$                                                                                           | $I^{D}$                                                                                                                       |
|--------|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| $S^{O}$ | **(1)**: traditional approach,  infection of susceptibles living in the same place as those infected | **(2)**: infected from other locations who arrived at the destination where will the infection be                                |
| $S^{D}$ | **(3)**: susceptibles who traveled and  were infected by infected people  living at the destination  | **(4)**: susceptible people went to a place  and there they found infected people  who live in another place but went  there too |

$$

\frac{dS_i}{dt} = - \beta \frac{m_{ii}}{N_i} S_i 
\frac{ \left ( \frac{ m_{ii} }{ N_i } \right ) I_i}{ 
\left ( \frac{ m_{ii} }{ N_i } \right ) N_i }, \;\; j=i, k=j=i

$$

$$ 
\frac{dS_i}{dt} = - \beta \frac{m_{ii}}{N_i} S_i
\frac{ \sum_{ \substack{k=1\\k\neq i}}^N  \left ( \frac{ m_{ki} }{ N_k } \right ) I_k}{ 
 \sum_{ \substack{k=1\\k\neq i}}^N \left ( \frac{ m_{ki} }{ N_k } \right ) N_k }, \;\; j=i

$$

$$

\frac{dS_i}{dt} = - \beta \sum_{ \substack{j=1\\j\neq i}}^N \left ( \frac{m_{ij}}{N_i} \right ) S_i \frac{ \left ( \frac{ m_{ij} }{ N_i } \right ) I_i}{ 
\left ( \frac{ m_{ij} }{ N_i } \right ) N_i }, \;\; k=i

$$

$$
\frac{dS_i}{dt} = -\beta \sum_{\substack{j=1\\j\neq i}}^N  
\left ( \frac{m_{ij}}{N_i} \right ) S_i 
\frac{ \sum_{\substack{k=1\\k\neq i}}^N  
\left ( \frac{m_{kj}}{N_k} \right ) I_k }{ \sum_{\substack{k=1\\k\neq i}}^N  
\left ( \frac{m_{ki}}{N_k} \right ) N_k }
$$

## Model for ODEINT - SIR Model for Metapopulation Networks 

$$

\frac{dS_i}{dt} = -\beta \sum_{\substack{j=1\\j\neq i}}^N  
\left ( \frac{m_{ij}}{N_i} \right ) S_i 
\frac{ \sum_{\substack{k=1\\k\neq i}}^N  
\left ( \frac{m_{kj}}{N_k} \right ) I_k }{ \sum_{\substack{k=1\\k\neq i}}^N  
\left ( \frac{m_{ki}}{N_k} \right ) N_k }
$$

$$

\frac{dI_i}{dt} = \beta \sum_{\substack{j=1\\j\neq i}}^N  
\left ( \frac{m_{ij}}{N_i} \right ) S_i 
\frac{ \sum_{\substack{k=1\\k\neq i}}^N  
\left ( \frac{m_{kj}}{N_k} \right ) I_k }{ \sum_{\substack{k=1\\k\neq i}}^N  
\left ( \frac{m_{ki}}{N_k} \right ) N_k }
- \gamma  I_i 

$$

\frac{dR_i}{dt} = \gamma I_i 

$$
