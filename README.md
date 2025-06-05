# RLE

Reinforcement Learning Exploration

## Reward Function

Weights are defined separately and the Reward function obtains metrics and scales them using the weights.

$$
W = \left[w_{goal}, w_{dist}, w_{align}, w_{collision}, w_{proximity}, w_{time}, w_{travel}\right]
$$

$$
R(t) = \underbrace{\left(R_{x}(t) + R_{\theta}(t)\right)}_{\text{Reaching Goal}} + \underbrace{\left(w_{dist} \Delta d\right)}_{\text{Distance Progress}} + \underbrace{\left(\begin{cases}w_{align} \cos(\theta_t) & \text{ if } d_t \geq \delta_p \\ 2w_{align}\cos(\Delta \psi_t) & \text{ if } d_t < \delta_p\end{cases}\right)}_{\text{Alignment Reward }} + \underbrace{\left(w_{collision}\cdot \mathcal{C}(t)\right)}_{\text{Collision Penalty}} + \underbrace{\left(w_{proximity}\cdot 1\{\rho_{min}(t) < \delta_{\rho}\}\right)}_{\text{Proxomity Penalty}} + \underbrace{\left(w_{time}\cdot t\right)}_{\text{Time Penalty}} + \underbrace{\left(w_{travel}\Delta d(t)\right)}_{\text{Travel Distance Penalty}}
$$

## Low-Pass Filter for smooth actions
$$
a_{s, t} = \alpha \cdot a_{r, t-1} + (1 - \alpha) \cdot a_{r, t}
$$