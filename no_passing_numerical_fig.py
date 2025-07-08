import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12

# Parameters
M0 = 10.0
v0 = 10000  
r0 = 2.5 * M0  
lambda0 = 50.0  

# Mass function and its derivative with smoothing near v0
def M(v):
    if v >= v0:
        return 0.0
    x = v/v0
    if x > 0.999:  # Smooth cutoff near v0
        return M0 * (0.001)**(1/3)
    return M0 * (1 - x)**(1/3)

def dM_dv(v):
    if v >= v0:
        return 0.0
    x = v/v0
    if x > 0.999:  # Cap the derivative
        return -M0/(3*v0) * (0.001)**(-2/3)
    return -M0/(3*v0) * (1 - x)**(-2/3)

# ODE system for solve_ivp (tau is first argument)
def geodesic_system(tau, y):
    v, r, lam = y
    
    if r <= 0.01:  # Stop near r=0
        return [0, 0, 0]
    
    M_v = M(v)
    dM_dv_v = dM_dv(v)
    
    # Cap lambda to prevent runaway
    lam = min(lam, 1000)
    
    dv_dtau = lam
    dr_dtau = 0.5 * (1/lam - (1 - 2*M_v/r)*lam)
    dlam_dtau = -(dM_dv_v/r) * lam**2 - (2*M_v/r**2) * lam * dr_dtau
    
    return [dv_dtau, dr_dtau, dlam_dtau]

# Event to stop when mass is nearly zero
def mass_zero_event(tau, y):
    return M(y[0]) - 1e-3 * M0

mass_zero_event.terminal = True
mass_zero_event.direction = -1

# Integration using solve_ivp
tau_span = (0, 200)
tau_eval = np.linspace(0, 200, 5000)
y0 = [0, r0, lambda0]

# Solve with adaptive stepping
sol = solve_ivp(geodesic_system, tau_span, y0, 
                t_eval=tau_eval, events=mass_zero_event,
                method='RK45', rtol=1e-8, atol=1e-10)

# Extract solution
tau = sol.t
v_sol = sol.y[0]
r_sol = sol.y[1]
lambda_sol = sol.y[2]

# Calculate epsilon and horizon position
M_sol = np.array([M(v) for v in v_sol])
r_horizon = 2 * M_sol
epsilon = r_sol - r_horizon

# Find when M < 0.1*M0
late_stage = M_sol < 0.1 * M0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True, constrained_layout=True)

# Top panel: r(tau) and horizon
ax1.plot(tau, r_sol, 'b-', linewidth=2, label='Infaller position $r(\\tau)$')
ax1.plot(tau, r_horizon, 'r--', linewidth=2, label='Horizon $r_h = 2M(v)$')
ax1.fill_between(tau, 0, r_horizon, alpha=0.2, color='red', label='Black hole interior')
ax1.set_ylabel('$r$ (geometric units)')
ax1.set_ylim(0, max(r0, max(r_sol)*1.1))
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_title('Infaller vs. Evaporating Horizon')

# Bottom panel: epsilon(tau)
ax2.plot(tau, epsilon, 'k-', linewidth=2, label='$\\epsilon(\\tau) = r - 2M(v)$')
ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
if any(late_stage):
    ax2.fill_between(tau[late_stage], 0, epsilon[late_stage], 
                     alpha=0.3, color='green', label='$M < 0.1M_0$ regime')
    ax2.legend(loc='upper left')
ax2.set_xlabel('Proper time $\\tau$')
ax2.set_ylabel('$\\epsilon(\\tau) = r - 2M(v)$')
ax2.set_ylim(-2, max(epsilon)*1.1)
ax2.grid(True, alpha=0.3)
ax2.set_title('Horizon Separation')

# Add inset to zoom in on the dramatic growth
if len(tau) > 200:
    # Create inset in upper right corner
    axins = inset_axes(ax2, width="40%", height="30%", loc='upper right', borderpad=3)    
    # Plot last 20% of data in the inset
    idx_zoom = int(0.8 * len(tau))
    axins.plot(tau[idx_zoom:], epsilon[idx_zoom:], 'k-', linewidth=2)
    axins.set_xlabel('$\\tau$', fontsize=10)
    axins.set_ylabel('$\\epsilon$', fontsize=10)
    axins.grid(True, alpha=0.3)
    axins.tick_params(labelsize=9)

# Add annotation at the right edge where epsilon is curving up
if len(epsilon) > 100:
    idx_annotate = -50  # Near the end of the data
    ax2.annotate('$\\epsilon$ grows rapidly\nas $M \\to 0$', 
                 xy=(tau[idx_annotate], epsilon[idx_annotate]),
                 xytext=(tau[idx_annotate]*0.85, epsilon[idx_annotate] - 0.3*max(epsilon)),
                 arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                 fontsize=11, ha='center')

plt.savefig('horizon_separation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('horizon_separation.png', dpi=300, bbox_inches='tight')
plt.show()

# Print some key values
print(f"Initial separation ε₀ = {epsilon[0]:.2f}")
print(f"Final separation ε_f = {epsilon[-1]:.2f}")
print(f"Initial mass M₀ = {M_sol[0]:.2f}")
print(f"Final mass M_f = {M_sol[-1]:.4f}")
print(f"Final v/v₀ = {v_sol[-1]/v0:.3f}")
print(f"Integration stopped at tau = {tau[-1]:.2f}")
if any(late_stage):
    print(f"When M = 0.1M₀, ε = {epsilon[late_stage][0]:.2f}")
