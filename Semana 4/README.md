<h1 align="center">
:mask:<br>Week 4
</h1>


---

# Simulating a viral attack on a social network using the SIR epidemiological model

The network captures 34 members of a karate club, documenting links between pairs of members who interacted outside the club.

**S**: The number of susceptible individuals. When a susceptible individual and an infectious individual come into "infectious contact", the susceptible individual contracts the disease and transitions into the infectious compartment. (Green nodes)

**I**: The number of infectious individuals. These are individuals who have been infected and are capable of infecting susceptible individuals. (Red nodes)

**R**: The number of removed (and immune) or deceased individuals. These are individuals who have been infected and have recovered from the disease and either entered the removed compartment or died. It is assumed that the number of deaths is insignificant in relation to the total population. This compartment can also be called "recovered" or "resistant". (Gray nodes)

## Zachary's Karate Club Network

When it comes to social network anlaysis, Zachary's Karate Club (ZKC) is perhaps the most well known example. It was introduced by Wayne Zachary in this paper in 1977 and has been a popular example ever since.

The network models the relationships between 34 members of a karate club: each node represents to an individual, and the links/edges represent individuals who interact outside of the karate club setting (e.g. spending social time together, like meeting up for a coffee, seperate to karate).

The network has two main players the 'Officer' - John A (node 33) and the instructor - Mr. Hi (node 0). And the story goes that a rift occurred between Mr Hi and John A, causing the karate club to splinter into two new clubs (or factions). One club lead by John A and the other led by Mr Hi.

One might expect that each member's decision to join either faction would be driven by their relationships with the other members of the club. So if we have a model of the relationships between the individuals (i.e. the network) we should be able to predict with faction each person will join. As you will see, network analysis gives us the power to do just that!

![Zachary's karate club](https://upload.wikimedia.org/wikipedia/en/thumb/8/87/Zachary_karate_club_social_network.png/1024px-Zachary_karate_club_social_network.png)

## Zachary's Methodology

Before the split each side tried to recruit adherents of the other party. Thus, communication flow had a special importance and the initial group would likely split at the "borders" of the network. Zachary used the maximum flow – minimum cut Ford–Fulkerson algorithm from “source” Mr. Hi to “sink” John A: the cut closest to Mr. Hi that cuts saturated edges divides the network into the two factions. Zachary correctly predicted each member's decision except member #9, who went with Mr. Hi instead of John A. 


## Dataset

The standard 78-edge network data set for Zachary's karate club is publicly available on the internet. The data can be summarized as list of integer pairs. Each integer represents one karate club member and a pair indicates the two members interacted. The data set is summarized below and also in the adjoining image. Node 1 stands for the instructor, node 34 for the club administrator / president.

```
[2 1]
[3 1] [3 2]
[4 1] [4 2] [4 3]
[5 1]
[6 1]
[7 1] [7 5] [7 6]
[8 1] [8 2] [8 3] [8 4]
[9 1] [9 3]
[10 3]
[11 1] [11 5] [11 6]
[12 1]
[13 1] [13 4]
[14 1] [14 2] [14 3] [14 4]
[17 6] [17 7]
[18 1] [18 2]
[20 1] [20 2]
[22 1] [22 2]
[26 24] [26 25]
[28 3] [28 24] [28 25]
[29 3]
[30 24] [30 27]
[31 2] [31 9]
[32 1] [32 25] [32 26] [32 29]
[33 3] [33 9] [33 15] [33 16] [33 19] [33 21] [33 23] [33 24] [33 30] [33 31] [33 32]
[34 9] [34 10] [34 14] [34 15] [34 16] [34 19] [34 20] [34 21] [34 23] [34 24] [34 27] [34 28] [34 29] [34 30] [34 31] [34 32] [34 33]
```

# Social Network Analysis

## Import the ZKC graph

Our first step is to import the ZKC graph. Thankfully, this is already included in networkx. We will create an instance of the graph and also retreive the club labels for each node. They tell us which faction of each member ends up siding with: Mr. Hi or the Officer (John A).

```
0 Mr. Hi
1 Mr. Hi
2 Mr. Hi
3 Mr. Hi
4 Mr. Hi
5 Mr. Hi
6 Mr. Hi
7 Mr. Hi
8 Mr. Hi
9 Officer
10 Mr. Hi
11 Mr. Hi
12 Mr. Hi
13 Mr. Hi
14 Officer
15 Officer
16 Mr. Hi
17 Mr. Hi
18 Officer
19 Mr. Hi
20 Officer
21 Mr. Hi
22 Officer
23 Officer
24 Officer
...
30 Officer
31 Officer
32 Officer
33 Officer
```
So we see, as expected, the members of the club are split between Mr. Hi's faction and the Officer's faction.

Mathmatically, a network is simply represented as an adjacency matrix, $A$. Each element of the matrix AijA_{ij}Aij​ represents the strength of the relationship between nodes i and j. Displaying AAA is one way for us to have a look at what is going on in the graph.

As a note on some of the definitions in graph theory relevant here: the ZKC graph is both undirected and unweighted. This means that the edges in the graph have no associated direction (so AAA is symmetric) and that the edges take a binary value of either 1 or 0 (i.e. the members either have a relationship outside of the karate club or not).

```
matrix([[0., 1., 1., ..., 1., 0., 0.],
        [1., 0., 1., ..., 0., 0., 0.],
        [1., 1., 0., ..., 0., 1., 0.],
        ...,
        [1., 0., 0., ..., 0., 1., 1.],
        [0., 0., 1., ..., 1., 0., 1.],
        [0., 0., 0., ..., 1., 1., 0.]])
```

## Visualise the graph we have just imported

| index | club_members |
|-------|--------------|
| 0     | Mr. Hi       |
| 1     | Mr. Hi       |
| 2     | Mr. Hi       |
| 3     | Mr. Hi       |
| 4     | Mr. Hi       |
| 5     | Mr. Hi       |
| 6     | Mr. Hi       |
| 7     | Mr. Hi       |
| 8     | Mr. Hi       |
| 9     | Officer      |
| 10    | Mr. Hi       |
| 11    | Mr. Hi       |
| 12    | Mr. Hi       |
| 13    | Mr. Hi       |
| 14    | Officer      |
| 15    | Officer      |
| 16    | Mr. Hi       |
| 17    | Mr. Hi       |
| 18    | Officer      |
| 19    | Mr. Hi       |
| 20    | Officer      |
| 21    | Mr. Hi       |
| 22    | Officer      |
| 23    | Officer      |
| 24    | Officer      |
| 25    | Officer      |
| 26    | Officer      |
| 27    | Officer      |
| 28    | Officer      |
| 29    | Officer      |
| 30    | Officer      |
| 31    | Officer      |
| 32    | Officer      |
| 33    | Officer      |

It becomes evident fairly quickly that the adjacency matrix is not the most intuitive way of visualising the karate club.

There are many methods we can use to visualise a network and here I will just focus on the functions built into networkx. They are based off of matplotlib and don't provide us with the prettiest visualisations you have ever seen, but they'll do the job here. Look out for my next article where I will explore graph visualisations in more detail!

```
<matplotlib.collections.PathCollection at 0x7f60ffe90f28>
```

**BEFORE:**

![ZKC_1](https://github.com/gabrielxcosta/Simulation-of-epidemiological-models-in-temporal-mobility-networks/blob/main/Semana%204/ZKC_1.png?raw=true)

**AFTER:**

![ZKC_2](https://github.com/gabrielxcosta/Simulation-of-epidemiological-models-in-temporal-mobility-networks/blob/main/Semana%204/ZKC_2.png?raw=true)

This is a lot more clear that the adjacency matrix! We quickly can see which members of the club are connected to eachother and we might even be able to draw a couple of quick initial conclusions. The first thing to note is that both Mr Hi and John A have the most connections in the graph (in other words they are the most central nodes). This is to be expected for the instructor and officer of the group. Given that we can see some nodes are connected to only one of Mr Hi or John A (but not the other) we could also start to make some educated guesses at which members will join each faction.

## Network Statistics (Exploratory Analysis)

Before we dive into some community detection it is worth exploring the network we have in a bit more detail (this is essentially our exploratory analysis stage). For networks, we often want to retreive some common network statistics.

The first statistic we will look at is the density. This is a measure of how complete the graph is (how many edges are present in the network as compared to the total possible number of edges).

```
The edge density is: 0.13903743315508021
```

This value of ~0.14 is roughly in line with what we might expect for a social network.

Next, let's look at the degree (how many edges each node has). This is a common centrality measure, which gives an idea of how 'imporant' each node is in the network. The assumption is that nodes with the most edges are the most important/central as they are directly connected to lots of other nodes. Nodes with a high centrality might be expected to play important roles in network. This is not the only way that centrality is measured in a graph and the interested reader is directed here.

```
The average degree is 4.588235294117647
```


```
Text(0.5, 1.0, 'Karate Club: Node Degree')
```
![DegreeDistribution](https://github.com/gabrielxcosta/Simulation-of-epidemiological-models-in-temporal-mobility-networks/blob/main/Semana%204/Karate_Club_Node_Degree.png)

This distribution is similar to what we might expect given the visualisation of the graph. The majority of members of the club do not have very many links (most have 2 or 3 links) but a few nodes (which in our case correspond to Mr Hi and John A) have a lot of links.

Another interesting statistic is the local clustering coefficient. You can find the full mathmatical definition here. The local clustering coefficient can be thought of as the average probability that a pair of node i’s friends are also friends with each other. In other words, it measures the extent to which any given node is located within a tight 'cluster' of neighbouring nodes.

![Clustering_SIR](https://github.com/gabrielxcosta/Simulation-of-epidemiological-models-in-temporal-mobility-networks/blob/main/Semana%204/Clustering_SIR.png?raw=true)

The clustering coefficient has been the subject of a lot of interesting research. A low value of the clustering coefficient indicates the presence of a 'strucutral hole', i.e. a location in the network where there are links missing that would have otherwise been expected. Which typically indicates that a node is located on the outside of a cluster. It has been argued that individuals at strucutral holes in the network are more likely to come up with good ideas because, being at the edge of a tight cluster of nodes, they are exposed to a greater variety of different ideas from outside of that cluster (Burt, 2004).

## Community Detection

A key aspect of network analysis is community detection. In a social network this is the idea that a large network can be broken down into smaller communinities/cliques. For example, if the network represents the social relationships of all the students at a school, a community/clique would be a friendship group.

The ability to detect communiites in networks has many applications. In the context of the karate club it will allow us to predict which members with side with Mr Hi and which will side with John A.

There are many ways to approach community detection in networks. I am not going to go into the maths in too much detail but we are going to opt for a modularity optimisation method. Modularity is a measure of the extent to which like is connected to like in a network. The algorithm we will chose is already implemented for us in networkx, which makes its implementation very easy!

```
3
```

```
[8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
[1, 2, 3, 7, 9, 12, 13, 17, 21]
[0, 4, 5, 6, 10, 11, 16, 19]
```

We immediately see a couple of interesting things in these communities. For instance, Mr. Hi (node 0) and John A (node 33) are in different communities. Given the split in the club this is what we would have expected - buts it's good to have it confirmed!

We can move on to visualise these different communities in the network. This will help to tell us how useful the communities we have detected really are. We are going to plot each communities in a different colour, and also include the label of which club each member ended up joining.

![Communities_1](https://github.com/gabrielxcosta/Simulation-of-epidemiological-models-in-temporal-mobility-networks/blob/main/Semana%204/Communities_1.png?raw=true)

From this we notice that the communitites detected actually match our split between the 'Officer' (John A) and Mr. Hi quite closely. Let's combine communities 1 (red) and 2 (blue) and take another look.

![Communities_2](https://github.com/gabrielxcosta/Simulation-of-epidemiological-models-in-temporal-mobility-networks/blob/main/Semana%204/Communities_2.png?raw=true)
 
 So, following the combination of those two communities we have split our network into two groups - the green group we predict to join John A's faction and the purple group Mr Hi's. Comparing to the club labels we see only one incorrect prediction in each group, meaning we have an accuracy of ~94%. Considering that that the analysis itself was straightforward to cary out, I'd say that is impressive!

This demonstrates the exciting power of network analysis. The idea that we can develop a mathmatical framework that can predict an individuals choices based off of their relationships with others is immensely powerful. We live in an interconnected world and the study of networks allows us to explore those connections.
 
## Wrap-Up

Hopefully this has opened your eyes to the exciting world of networks! The uses of this type of analysis stretch far and wide, and are rapidly increasing in popularity. There are many different forms that network analysis can take, this article just scratches the surface. A whole host of different algorithms are implemented in networkx. You can open up this notebook in deepnote and have a play around building on the analysis already performed here!
 
## Results

### Nodes Degrees

|Nodes  |Degrees|
|-------|-------|
|Node 0 |16     |
|Node 1 |9      |
|Node 2 |10     |
|Node 3 |6      |
|Node 4 |3      |
|Node 5 |4      |
|Node 6 |4      |
|Node 7 |4      |
|Node 8 |5      |
|Node 9 |2      |
|Node 10|3      |
|Node 11|1      |
|Node 12|2      |
|Node 13|5      |
|Node 14|2      |
|Node 15|2      |
|Node 16|2      |
|Node 17|2      |
|Node 18|2      |
|Node 19|3      |
|Node 20|2      |
|Node 21|2      |
|Node 22|2      |
|Node 23|5      |
|Node 24|3      |
|Node 25|3      |
|Node 26|2      |
|Node 27|4      |
|Node 28|3      |
|Node 29|4      |
|Node 30|4      |
|Node 31|6      |
|Node 32|12     |
|Node 33|17     |

### Running the 1st Simulation Picking One Random Initial Node

## $$\beta = 0.6$$

## $$\gamma = 0.4$$

![Simulation](https://github.com/gabrielxcosta/Simulation-of-epidemiological-models-in-temporal-mobility-networks/blob/main/Semana%204/Timelapse%201/sir_spreading.gif?raw=true)

### Running the 2st Simulation Picking One Random Initial Node with SIR Model Chart

## $$\beta = 0.7$$

## $$\gamma = 0.4$$

![Simulation2](https://github.com/gabrielxcosta/Simulation-of-epidemiological-models-in-temporal-mobility-networks/blob/main/Semana%204/Timelapse%202/sir_model_2.gif?raw=true)

![Simulation2 - StaticChart](https://github.com/gabrielxcosta/Simulation-of-epidemiological-models-in-temporal-mobility-networks/blob/main/Semana%204/Timelapse%202/SIR_In_Time.png?raw=true)
