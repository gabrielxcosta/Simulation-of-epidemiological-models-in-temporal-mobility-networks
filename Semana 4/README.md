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

![Simulation](https://github.com/gabrielxcosta/Simulation-of-epidemiological-models-in-temporal-mobility-networks/blob/main/Semana%204/Templates/sir_spreading.gif?raw=true)

### Running the 2st Simulation Picking One Random Initial Node with SIR Model Chart

![Simulation2](https://github.com/gabrielxcosta/Simulation-of-epidemiological-models-in-temporal-mobility-networks/blob/main/Semana%204/Timelapse%202/sir_model_2.gif?raw=true)

![Simulation2 - StaticChart](https://github.com/gabrielxcosta/Simulation-of-epidemiological-models-in-temporal-mobility-networks/blob/main/Semana%204/Timelapse%202/SIR_In_Time.png?raw=true)
