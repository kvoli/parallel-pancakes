# parallel-pancakes
solution to diameter of a n-pancake graph using mpi. Achieves linear speedup w.r.t processors used.

## Abstract

>We are given a stack of pancakes of different sizes and are only allowed to flip some number of pancakes from the top. Solving the maximum number of pancake flips one would need in order to sort any arbitrary stack is equivalent to finding the diameter of a pancake graph. The diameter of a pancake graph can be computed as a single source shortest path from one vertex to every other vertex in the graph. Finding the diameter of an n-pancake graph is a \textbf{very hard} problem and has only been solved up to $N=19$. In this paper we propose a simple by design parallelization approach to solve for the diameter of an n-pancake graph. Our approach, implemented using OpenMPI achieves a linear speedup w.r.t processors used.

## Results

![pancake_reuslts](https://imgur.com/vIqq0RV)

## Method

Parallel A* with message passing using MPI. Reducing search space dramatically by Heydari's findings for calculating only n, n-1 and n-2.
