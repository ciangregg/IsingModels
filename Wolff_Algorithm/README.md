# Animations of the Wolff Algorithm

## Cluster growth 
First, we for lattice size $20\times20$ with LHS $\beta=0.40$ and RHS $\beta=0.50$
<table>
  <tr>
    <td><img src="plots/cluster_growth/wolff_cluster_growth_20x20_and_beta0.4.gif" width="400"></td>
    <td><img src="plots/cluster_growth/wolff_cluster_growth_20x20_and_beta0.5.gif" width="400"></td>
  </tr>
</table>
Next, we for lattice size $50\times50$ with LHS $\beta=0.40$ and RHS $\beta=0.50$

<table>
  <tr>
    <td><img src="plots/cluster_growth/wolff_cluster_growth_50x50_and_beta0.4.gif" width="400"></td>
    <td><img src="plots/cluster_growth/wolff_cluster_growth_50x50_and_beta0.5.gif" width="400"></td>
  </tr>
</table>
Lastly, we for lattice size $100\times100$ with LHS $\beta=0.40$ and RHS $\beta=0.50$

<table>
  <tr>
    <td><img src="plots/cluster_growth/wolff_cluster_growth_100x100_and_beta0.4.gif" width="400"></td>
    <td><img src="plots/cluster_growth/wolff_cluster_growth_100x100_and_beta0.5.gif" width="400"></td>
  </tr>
</table>
We tried $1000\times1000$ however it crashed my vscode when animating due to memory overload.

## Cluster growth over spin state

We've plotted our ising spin state with $+1$ in grey and $-1$ in black, with the cluster spreading in purple. 

In an ordered states $<T_c$

<img src="plots/growth_plus_spin/wolff_growth_100x100_beta0.5.gif" width="800px">

In a disordered states $<T_c$

<img src="plots/growth_plus_spin/wolff_growth_100x100_beta0.49.gif" width="800px">

<img src="plots/growth_plus_spin/wolff_growth_100x100_beta0.48.gif" width="800px">

