import numpy as np
import scipy as sp
import matplotlib.pyplot as plot
import matplotlib.animation as animation

def replace_vars(string, dict):
    for key,value in dict.items():
        string = string.replace('{{'+key+'}}', str(value))
    return string

N = 1000
SIZE = 10
dt = SIZE/N

def time(t):
    return round(t/SIZE * N)

m = 1
w = 25
# some changes of variables
k = dt*dt*w*w*0.25
u_to_x_fac = 1/np.sqrt(1+k)/np.sqrt(m/dt)
g = (1-k)/(1+k)
#S = np.sum(u[1:-1]**2) - g*np.sum(u[:-1]*u[1:])


N_SWEEPS = 100
MONTE_CARLO_N_ROUNDS = 3
MONTE_CARLO_WORKER_COUNT = 1000
SHAPE = (MONTE_CARLO_WORKER_COUNT, N_SWEEPS, N)

# GPU CODE =============================
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

prg = cl.Program(ctx, replace_vars("""
__kernel void sweep(
    __global const float *deltas_g, __global const float *probs_g, __global float *u_global_g, __global int *acceptance_rate_g)
{
    // initialize variables
    int gid = get_global_id(0); // each gid will get its own u to play with - these will be added up later
    float delta, dS;
    int accept;
    int I, u_I;
  
    for (int sweep_j = 0; sweep_j < {{VAR_SHAPE1}}; sweep_j++){
        for (int u_i = 0; u_i < {{VAR_SHAPE2}}; u_i++){
            I = u_i + sweep_j*{{VAR_SHAPE2}} + gid*{{VAR_SHAPE2}}*{{VAR_SHAPE1}};
            u_I = u_i + gid*{{VAR_N}};
            
            delta = deltas_g[I]; // fetch delta and probability from CPU-produced array
          
            // compute dS and accept?
            dS = delta * (delta + 2 * u_global_g[u_I] - {{VAR_G}} * (u_global_g[u_I-1] + u_global_g[u_I+1]));
            accept = probs_g[I] < exp(-dS);
            acceptance_rate_g[gid] += accept;
          
            // replace value in u, sequentially as we go, for all the sweeps
            u_global_g[u_I] += accept*delta;
        }
    }
}
""", {
    'VAR_G': g,
    'VAR_N': N,
    'VAR_SWEEPS': N_SWEEPS,
    'VAR_SHAPE2': SHAPE[2],
    'VAR_SHAPE1': SHAPE[1],
    'VAR_SHAPE0': SHAPE[0],
})).build()

# GPU CODE =============================

# OBSERVATIONS =========================
u = np.empty((MONTE_CARLO_WORKER_COUNT, N), dtype=np.float32)
u_global_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=u)
DELTA = 1.8
OBS = []

for n in range(MONTE_CARLO_N_ROUNDS):
    print('Run %s...' % n)
    cl.enqueue_copy(queue, u_global_g, np.zeros((MONTE_CARLO_WORKER_COUNT, N), dtype=np.float32))

    deltas = 2 * DELTA * (np.random.random(SHAPE) - 0.5).astype(np.float32)
    probs = np.random.random(SHAPE).astype(np.float32)
    acceptance_rate = np.zeros(MONTE_CARLO_WORKER_COUNT, dtype=np.int32)
    #deltas_g = pyopencl.array.to_device(ctx, deltas)
    #probs_g = pyopencl.array.to_device(ctx, probs)
    deltas_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=deltas)
    probs_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=probs)
    acceptance_rate_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=acceptance_rate)

    print(f"Total memory transfer: {round((deltas_g.size + probs_g.size + u_global_g.size + acceptance_rate_g.size)/1000000, 2)} MB")

    prg.sweep(queue, (MONTE_CARLO_WORKER_COUNT,), None, deltas_g, probs_g, u_global_g, acceptance_rate_g)

    deltas_g.release()
    probs_g.release()
    cl.enqueue_copy(queue, u, u_global_g)
    cl.enqueue_copy(queue, acceptance_rate, acceptance_rate_g)

    print('Mean acceptance rate:', np.round(np.mean(acceptance_rate).astype(float)/N/N_SWEEPS, 2))

    #f, pl = plot.subplots(4, 1)
    #i = 0

    for U in u:
        #OBS.append(U[time(4.0):time(5.1)])
        """
        pl[i].plot(np.linspace(0, SIZE, N), U)
        plot.xlabel('t')
        pl[i].set_ylabel('x')
        if i != 3:
            pl[i].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)
        pl[i].set_ylim([-5, 5])
        i += 1
        if i == 4:
            plot.show()
            exit()
        """
        OBS.append(U[time(4.95)]*U[time(4.95):time(5.1)])

print(f'Done! N={len(OBS)} observations collected')
MEAN_OBS = np.mean(np.array(OBS), axis=0)
# OBSERVATIONS =========================

# ERROR CALC ===========================
def R(h, obs, obs_expected):
    N_obs = len(obs)
    return (sum(((obs[i] - obs_expected)*(obs[i+h] - obs_expected)) for i in range(0, N_obs-h))/(N_obs-h))

print('Calculating error...')
N_obs = len(OBS)
#tau = R(0, OBS, MEAN_OBS) + 2*sum(R(h, OBS, MEAN_OBS) for h in range(1, N_obs))
#N_obs_eff = N_obs/2/tau
err = np.sqrt(np.abs((R(0, OBS, MEAN_OBS) + 2*sum(R(h, OBS, MEAN_OBS) for h in range(1, N_obs)))) / len(OBS))
#err = np.sqrt(np.abs(1/N_obs_eff))
print('Graphing...')

pass
# ERROR CALC ===========================

#plot.plot(np.arange(0,MONTE_CARLO_N), OBS/np.arange(1,MONTE_CARLO_N+1))
plot.errorbar(np.linspace(0, 0.15/dt, len(MEAN_OBS)), MEAN_OBS*u_to_x_fac**2, color='#00f', yerr=err*u_to_x_fac**2, capthick=1, capsize=3, ecolor='#f00')
plot.xlabel('t/dt')
plot.ylabel('<x(4.95+t)x(4.95)>')
plot.title('Harmonic Oscillator <xx> from t=4.95 to t=5.10')
plot.show()