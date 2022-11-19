<h1 align="center">
📄<br>Week 4
</h1>


---

## Zachary's Karate Club Network

Zachary's karate club is a social network of a university karate club, described in the paper "An Information Flow Model for Conflict and Fission in Small Groups" by Wayne W. Zachary. The network became a popular example of community structure in networks after its use by Michelle Girvan and Mark Newman in 2002.

[![Zachary's karate club](https://en.wikipedia.org/wiki/Zachary%27s_karate_club)

## Zachary's Methodology

Before the split each side tried to recruit adherents of the other party. Thus, communication flow had a special importance and the initial group would likely split at the "borders" of the network. Zachary used the maximum flow – minimum cut Ford–Fulkerson algorithm from “source” Mr. Hi to “sink” John A: the cut closest to Mr. Hi that cuts saturated edges divides the network into the two factions. Zachary correctly predicted each member's decision except member #9, who went with Mr. Hi instead of John A. 


## Dataset

The standard 78-edge network data set for Zachary's karate club is publicly available on the internet.[3] The data can be summarized as list of integer pairs. Each integer represents one karate club member and a pair indicates the two members interacted. The data set is summarized below and also in the adjoining image. Node 1 stands for the instructor, node 34 for the club administrator / president.

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
