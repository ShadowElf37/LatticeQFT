import numpy as np
import scipy as sp
import matplotlib.pyplot as plot
import matplotlib.animation as animation

N = 1000
SIZE = 10

x = np.linspace(-SIZE//2, SIZE//2, N)
t = np.linspace(0, SIZE, N)

def time(t):
    return round(t/SIZE * N)

m = 1
w = 25
dt = SIZE/N

# some changes of variables
k = dt*dt*w*w*0.25
u = np.sqrt(1+k) * np.sqrt(m/dt) * x
g = (1-k)/(1+k)

#S = np.sum(u[1:-1]**2) - g*np.sum(u[:-1]*u[1:])

print(w*dt)

# GPU CODE =============================
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

prg = cl.Program(ctx, """
__kernel void sweep(
    __global const int *indices_g, __global const float *deltas_g, __global float *probs_g, __global float *u_global_g)
{
  int gid = get_global_id(0);
  
  int i = indices_g[gid];
  float delta = deltas_g[gid];
  float prob = probs_g[gid];
  
  float dS = delta * (delta + 2 * u_global_g[i] - VAR_G * (u_global_g[i-1] + u_global_g[i+1]));
  int accept = prob < exp(-dS);
  
  u_global_g[i] = u_global_g[i] + accept*delta;
}
""".replace('VAR_G', str(g))).build()

# GPU CODE =============================

N_sweeps = 100
MONTE_CARLO_N = 2000
OBS = 0

SWEEP_N = N
DELTA = 1.5
u0 = np.zeros(N)
u_global = u0.copy()
u_global_g = cl.Buffer(ctx, mf.READ_ONLY, u0.nbytes)
u_new_g = cl.Buffer(ctx, mf.WRITE_ONLY, u0.nbytes)

for n in range(MONTE_CARLO_N):
    cl.enqueue_copy(queue, u_global_g, u0)

    for _ in range(N_sweeps):
        indices = np.round(np.random.random(SWEEP_N) * (len(u) - 3)).astype(np.int32) + 1
        deltas = 2 * DELTA * (np.random.random(SWEEP_N) - 0.5).astype(np.float32)
        probs = np.random.random(SWEEP_N).astype(np.float32)
        indices_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indices)
        deltas_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=deltas)
        probs_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=probs)

        prg.sweep(queue, indices.shape, None, indices_g, deltas_g, probs_g, u_global_g)
        #cl.enqueue_copy(queue, u_global_g, u_new_g)

    cl.enqueue_copy(queue, u_global, u_global_g)

    OBS = u_global[time(4.95)]*u_global[time(4.95):time(5.05)] + OBS

#plot.plot(np.arange(0,MONTE_CARLO_N), OBS/np.arange(1,MONTE_CARLO_N+1))
plot.plot(np.linspace(0, 0.4/dt, len(OBS)), OBS/MONTE_CARLO_N)
plot.show()