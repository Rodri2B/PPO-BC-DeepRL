import os
import numpy as np
import casadi as ca
import time
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
import matplotlib.pyplot as plt

# ==========================================
# 1. Define the Nonlinear Model (CasADi)
# ==========================================
def export_cart_pendulum_model() -> AcadosModel:
    model_name = 'cart_pendulum'
    m1, m2, l, g = 1.0, 2.0, 1.0, 9.81

    # Define symbolic states
    p = ca.SX.sym('p')
    theta = ca.SX.sym('theta')
    v = ca.SX.sym('v')
    omega = ca.SX.sym('omega')
    x = ca.vertcat(p, theta, v, omega)

    # Define symbolic inputs
    u = ca.SX.sym('u')

    # Define state derivatives (for implicit formulation)
    xdot = ca.SX.sym('xdot', 4)

    # Exact continuous-time nonlinear dynamics
    sin_t = ca.sin(theta)
    cos_t = ca.cos(theta)
    denom = m1 + m2 - m2 * cos_t**2

    p_dot = v
    theta_dot = omega
    v_dot = (u + m2 * l * omega**2 * sin_t - m2 * g * sin_t * cos_t) / denom
    omega_dot = ((m1 + m2) * g * sin_t - cos_t * (u + m2 * l * omega**2 * sin_t)) / (l * denom)

    f_expl = ca.vertcat(p_dot, theta_dot, v_dot, omega_dot)
    f_impl = xdot - f_expl

    # Populate the acados model
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    return model

# ==========================================
# 2. Setup the Optimal Control Problem (OCP)
# ==========================================
def setup_nmpc():
    ocp = AcadosOcp()
    model = export_cart_pendulum_model()
    ocp.model = model

    nx = model.x.shape[0]
    nu = model.u.shape[0]
    ny = nx + nu  # Track all states and inputs in the stage cost
    ny_e = nx     # Track only states in the terminal cost

    # Horizon and timing
    N = 40
    dt = 0.1
    Tf = N * dt
    ocp.dims.N = N

    # --- Cost Function (Linear Least Squares) ---
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    # Weights: penalize [p, theta, v, omega, u]
    Q = np.diag([10.0, 10.0, 0.1, 0.1])
    R = np.diag([0.01])
    ocp.cost.W = np.block([[Q, np.zeros((nx, nu))], [np.zeros((nu, nx)), R]])
    ocp.cost.W_e = Q

    # Mapping states/controls to the stage cost residual
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:, :nu] = np.eye(nu)

    # Mapping states to terminal cost residual
    ocp.cost.Vx_e = np.eye(nx)

    # Default references (will be updated in loop)
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # --- Constraints ---
    # Input bounds: u between -15.0 and 15.0
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbu = np.array([-15.0])
    ocp.constraints.ubu = np.array([15.0])

    # State bounds: p (index 0) constrained between -1.5 and 1.5 (The Wall Constraint)
    ocp.constraints.idxbx = np.array([0])
    ocp.constraints.lbx = np.array([-1.5])
    ocp.constraints.ubx = np.array([1.5])

    # Terminal state bounds (same as above)
    ocp.constraints.idxbx_e = np.array([0])
    ocp.constraints.lbx_e = np.array([-1.5])
    ocp.constraints.ubx_e = np.array([1.5])

    # Initial state
    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0])

    # --- Solver Options ---
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK' # Explicit Runge-Kutta 4
    
    # THE MAGIC FOR REAL-TIME: Real-Time Iteration (SQP_RTI)
    # Does exactly 1 QP solve per timestep. Ultra-fast and deterministic.
    ocp.solver_options.nlp_solver_type = 'SQP_RTI' 
    ocp.solver_options.tf = Tf

    # Generate and compile C code
    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    sim_solver = AcadosSimSolver(ocp, json_file='acados_ocp.json') # For simulating the plant

    return ocp_solver, sim_solver, nx, nu

# ==========================================
# 3. Main Real-Time Loop
# ==========================================
def run_realtime_nmpc():
    ocp_solver, sim_solver, nx, nu = setup_nmpc()

    N_steps = 200
    N_horizon = 40
    dt = 0.1

    # Arrays to store data
    x_hist = np.zeros((nx, N_steps + 1))
    u_hist = np.zeros((nu, N_steps))
    solve_times = np.zeros(N_steps)

    x_current = np.array([0.0, 0.0, 0.0, 0.0])
    x_hist[:, 0] = x_current

    print("\nStarting Real-Time Loop...")
    for i in range(N_steps):
        # 1. Provide current state feedback to the solver
        ocp_solver.set(0, "lbx", x_current)
        ocp_solver.set(0, "ubx", x_current)

        # 2. Update Reference Trajectory for the horizon
        for k in range(N_horizon):
            t_ref = (i + k) * dt
            # Step reference to pos=2.0 at t > 2s
            pos_ref = 2.0 if t_ref > 2.0 else 0.0
            
            yref = np.array([pos_ref, 0.0, 0.0, 0.0, 0.0]) # [p, theta, v, omega, u]
            ocp_solver.set(k, "yref", yref)

        # Terminal node reference
        t_ref_e = (i + N_horizon) * dt
        pos_ref_e = 2.0 if t_ref_e > 2.0 else 0.0
        ocp_solver.set(N_horizon, "yref_e", np.array([pos_ref_e, 0.0, 0.0, 0.0]))

        # 3. SOLVE OCP (This runs the compiled C-code)
        t_start = time.perf_counter()
        status = ocp_solver.solve()
        solve_times[i] = (time.perf_counter() - t_start) * 1000.0 # to milliseconds

        if status != 0:
            print(f"Solver failed at step {i} with status {status}")

        # 4. Extract control action
        u0 = ocp_solver.get(0, "u")
        u_hist[:, i] = u0

        # 5. Simulate real plant (stepping the exact nonlinear physics forward)
        sim_solver.set("x", x_current)
        sim_solver.set("u", u0)
        sim_solver.solve()
        x_current = sim_solver.get("x")
        
        x_hist[:, i + 1] = x_current

    print(f"Finished. Average solve time: {np.mean(solve_times):.3f} ms")
    print(f"Max solve time: {np.max(solve_times):.3f} ms")

    # Plotting
    tspan = np.arange(0, N_steps * dt, dt)
    plt.figure(figsize=(10, 8))

    plt.subplot(311)
    plt.plot(tspan, [2.0 if t>2.0 else 0.0 for t in tspan], 'r--', label='Ref')
    plt.plot(tspan, x_hist[0, :-1], 'k', label='Pos Actual')
    plt.axhline(1.5, color='orange', linestyle=':', label='Wall Constraint')
    plt.ylabel('Pos [m]')
    plt.legend()
    plt.grid()

    plt.subplot(312)
    plt.plot(tspan, x_hist[1, :-1], 'g', label='Angle Actual')
    plt.ylabel('Angle [rad]')
    plt.grid()

    plt.subplot(313)
    plt.plot(tspan, u_hist[0, :], 'b')
    plt.ylabel('Force [N]')
    plt.xlabel('Time [s]')
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_realtime_nmpc()