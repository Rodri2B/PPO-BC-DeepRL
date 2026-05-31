import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Exact Nonlinear System Dynamics
# ==========================================
m1, m2, l, g = 1.0, 2.0, 1.0, 9.81
nx, nu, ny = 4, 1, 2

# Symbolic variables for CasADi
x = ca.MX.sym('x', nx)
u = ca.MX.sym('u', nu)

p, theta, v, omega = x[0], x[1], x[2], x[3]

# Exact nonlinear equations of motion for Pendulum-Cart
sin_t = ca.sin(theta)
cos_t = ca.cos(theta)
denom = m1 + m2 - m2 * cos_t**2

p_dot = v
theta_dot = omega
v_dot = (u + m2 * l * omega**2 * sin_t - m2 * g * sin_t * cos_t) / denom
omega_dot = ((m1 + m2) * g * sin_t - cos_t * (u + m2 * l * omega**2 * sin_t)) / (l * denom)

# Continuous time nonlinear dynamics function: x_dot = f(x, u)
x_dot = ca.vertcat(p_dot, theta_dot, v_dot, omega_dot)
f_cont = ca.Function('f_cont', [x, u], [x_dot])

# ==========================================
# 2. RK4 Discretization Function
# ==========================================
dt = 0.1
k1 = f_cont(x, u)
k2 = f_cont(x + dt/2 * k1, u)
k3 = f_cont(x + dt/2 * k2, u)
k4 = f_cont(x + dt * k3, u)
x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Discrete time nonlinear dynamics function: x_k+1 = F(x_k, u_k)
F_RK4 = ca.Function('F_RK4', [x, u], [x_next])

# ==========================================
# 3. NMPC Setup with Opti Stack
# ==========================================
N = 40  # Prediction horizon

opti = ca.Opti()

## Configure IPOPT for Real-Time execution
#opts = {
#    'ipopt.print_level': 0,
#    'print_time': 0,
#    'ipopt.sb': 'yes',
#    
#    # --- REAL-TIME SETTINGS ---
#    'ipopt.max_iter': 15,          # Strictly cap the number of iterations
#    'ipopt.max_cpu_time': 0.05,    # Hard limit of 50ms (well below your 100ms dt)
#    'ipopt.tol': 1e-3,             # Slightly relax tolerance to find solutions faster
#    
#    # Optional: Just-In-Time (JIT) compilation to C-code for massive speedups
#    # (Requires a C compiler like gcc/clang installed on your system)
#    'jit': True,
#    'compiler': 'shell',
#    'jit_options': {'flags': ['-O3']} 
#}
#
#opti.solver('ipopt', opts)

# Decision Variables (Multiple Shooting)
X = opti.variable(nx, N + 1)
U = opti.variable(nu, N)

# Parameters (Allows us to update the problem rapidly in a loop without rebuilding)
x0_param = opti.parameter(nx)       # Current state feedback
y_ref_param = opti.parameter(ny, N) # Reference trajectory over horizon

# Cost weights
Q = ca.diag([10.0, 10.0]) # Weights for [theta, p]
R = 0.01

cost = 0

# Initial condition constraint
opti.subject_to(X[:, 0] == x0_param)

for k in range(N):
    # Multiple Shooting constraint: enforce physics
    opti.subject_to(X[:, k+1] == F_RK4(X[:, k], U[:, k]))
    
    # ----------------------------------------------------
    # OBSERVED STATE CONSTRAINTS (y = h(x))
    # ----------------------------------------------------
    # Let's say we observe Angle (theta) and Position (p)
    y_k = ca.vertcat(X[1, k], X[0, k]) 
    
    # Let's constrain the cart position so it doesn't hit a wall at p = 1.5 and p = -1.5
    # (Even if the reference asks it to go further, the constraint will stop it)
    opti.subject_to(opti.bounded(-1.5, X[0, k], 1.5)) 
    
    # Input constraints
    opti.subject_to(opti.bounded(-15.0, U[:, k], 15.0))
    
    # Stage cost
    err = y_k - y_ref_param[:, k]
    cost += ca.mtimes([err.T, Q, err]) + R * U[:, k]**2

opti.minimize(cost)

# Configure IPOPT Solver (Hide massive print outputs for the loop)
opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
opti.solver('ipopt', opts)

# ==========================================
# 4. Simulation Loop
# ==========================================
tend = 20.0
tspan = np.arange(0, tend + dt, dt)
looplen = len(tspan)

# Reference generation (Asking the cart to go to p=2.0, which violates our 1.5 wall constraint!)
r_base = np.zeros((ny, looplen + N))
for k in range(looplen + N):
    t = k * dt
    if t <= 2: r_base[:, k] = [0, 0]
    elif t <= 10: r_base[:, k] = [0, 2.0] # Angle 0, Pos 2.0
    else: r_base[:, k] = [0, 0]

x0 = np.array([0.0, 0.0, 0.0, 0.0]) # [p, theta, v, omega]
x_hist = np.zeros((nx, looplen))
u_hist = np.zeros((nu, looplen))

for i in range(looplen):
    # Update parameters for the current step
    opti.set_value(x0_param, x0)
    opti.set_value(y_ref_param, r_base[:, i:i+N])
    
    try:
        # Solve the NLP
        sol = opti.solve()
        u_opt = sol.value(U[:, 0]) # Extract first control action
        
        # Pass current solution as initial guess for next step (Warm Starting)
        opti.set_initial(X, sol.value(X))
        opti.set_initial(U, sol.value(U))
    except RuntimeError:
        # If solver fails (e.g., infeasible), extract best guess
        print(f"Solver failed at step {i}")
        u_opt = opti.debug.value(U[:, 0])

    # Store data
    x_hist[:, i] = x0
    u_hist[:, i] = u_opt
    
    # Simulate real system exactly using the RK4 function
    x0 = np.array(F_RK4(x0, u_opt)).flatten()

# ==========================================
# 5. Plot Results
# ==========================================
plt.figure(figsize=(10, 8))

plt.subplot(311)
plt.plot(tspan, r_base[1, :looplen], 'r--', label='Pos Ref (2.0)')
plt.plot(tspan, x_hist[0, :], 'k-', linewidth=2, label='Pos Actual (p)')
plt.axhline(1.5, color='orange', linestyle=':', linewidth=2, label='Wall Constraint (1.5)')
plt.ylabel('Cart Position [m]')
plt.legend(loc='lower right')
plt.grid()

plt.subplot(312)
plt.plot(tspan, r_base[0, :looplen], 'b--', label='Angle Ref')
plt.plot(tspan, x_hist[1, :], 'g-', label='Angle Actual (θ)')
plt.ylabel('Pendulum Angle [rad]')
plt.legend()
plt.grid()

plt.subplot(313)
plt.plot(tspan, u_hist[0, :], 'r-')
plt.ylabel('Control Force [N]')
plt.xlabel('Time [s]')
plt.grid()
plt.tight_layout()
plt.show()