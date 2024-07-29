# Training Algorithm

## Value and Policy Iteration 

### Getting Trajectory

tau = (s_0, a_0, s_1, a_1, ...)

### Modelling Rewards and Return

r_t = R(s_t, a_t, s_{t+1})

R(tau) = sum_{t=0}^{T}(r_t)

G(tau) = sum_{t=0}^{T}(r_t * (gamma^t))

### State Value Targets at Each State

V^pi(s) = E_{tau ~ pi} [G(tau) | s_0 = s]

### State Action Value Targets at each state

Q^pi(s,a) = E_{tau ~ pi}[G(tau)| s_0 = s, a = a]

### Policy targets at each state

pi(s,a) = argmax_a(Q(s,a))

### State Action update (undiscounted)

Q(s_t,a_t) = r_t * Q(s_t+1,a_t+1)

