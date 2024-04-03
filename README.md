Algorithmic Realization: Solution to the Sign Conflict Problem for Hanging Nodes on Hp-Hexahedral Nédélec Elements 
================ 
 
This GitHub repository contains the implementation of an algorithmic solution to the sign conflict problem 
for hanging nodes on hp-hexahedral Nédélec elements. The starting point for this work was DEALII 9.5.2. 
 
# Overview 
The sign conflict problem arises for Nédélec elements, as Nédélec elements are oriented. 
These elements are widely used in electromagnetic simulations and other scientific computing applications. 
Especially in the presence of hanging faces, one has to take special care of the orientation. 
The algorithmic realization presented here addresses the orientation problem in the presence of hanging faces 
and provides a robust solution. 
 
The support for hanging nodes for Nedelec elements is already included in the deal.II master, and will be part of the next release. 
 
# Preprint 
For a detailed description of the implementation, please refer to our preprint: 
[Algorithmic Realization of the Solution to the Sign Conflict Problem for Hanging Nodes on Hp-Hexahedral Nédélec Elements](https://arxiv.org/abs/2306.01416)
