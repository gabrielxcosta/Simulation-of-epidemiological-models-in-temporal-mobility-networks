# Metapopulation Models for Disease Transmission

This project aims to simulate disease transmission in a network of geographically isolated populations of hosts using metapopulation models. The models account for the local conditions that may affect transmission intensity in each population.

## Overview

Each population $i$ contains $N_{i}$ hosts. Disease transmission is assumed to be completely local, meaning that hosts from different subpopulations come into contact with one another only if they travel to occupy the same location. The project describes and compares two simple models of host movement, each of which represents a different set of rules governing how hosts move from one location to another.

## Flux Model

The Flux model is a **Eulerian movement model** that describes hosts as diffusing from one metapopulation to another. The following differential equation represents the model:

$$\frac{dN_{i}}{dt} = -\sum_{j=1}^{K} f_{i, j} N_{i} + \sum_{j=1}^{K} f_{j, i} N_{j}$$ 

where $N_{i}$ counts the number of hosts currently located at site $i$, and $K$ is the total number of populations. The total number of hosts remains constant over time ($N = \sum_{1}^{K} N_{i}$). The constant $f_{i, j}$ represents the rate at which hosts located at $i$ travel to $j$, where $f_{i, i}$ for all $i$. The fully specified Flux model requires $K(K - 1)$ parameters.

## SIR Model

Combining the basic SIR model with the Flux movement model, we obtain an analogous set of $3K$ equations:

$$\frac{dS_{i}}{dt} = -\beta_{i} \frac{S_{i}I_{i}}{N_{i}} -\sum_{j=1}^{K} f_{i, j} S_{i} + \sum_{j=1}^{K} f_{j, i} S_{j}$$

$$\frac{dI_{i}}{dt} = \beta_{i} \frac{S_{i}I_{i}}{N_{i}} - \gamma I_{i} -\sum_{j=1}^{K} f_{i, j} I_{i} + \sum_{j=1}^{K} f_{j, i} I_{j}$$

$$\frac{dR_{i}}{dt} = \gamma I_{i} -\sum_{j=1}^{K} f_{i, j} R_{i} + \sum_{j=1}^{K} f_{j, i} R_{j}$$

## Usage

The project is implemented in Python, and the code is available in the ``metaSIR.py`` file. The project can be used to simulate disease transmission in a network of geographically isolated populations by initializing the necessary parameters and running the simulation.
