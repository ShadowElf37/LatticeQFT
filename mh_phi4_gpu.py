import numpy as np
import matplotlib.pyplot as plot
import matplotlib.animation as animation
import time

def render(string):
    import re
    for kw in set(re.findall(r'{{(.[^\}]*)}}', string)):
        string = string.replace('{{' + kw + '}}', str(eval(kw)))
    return string

Nt = 50
Nx = 50
Ny = 50

def index(t, x, y):
    #all coords go from 0 to 1
    return round(t * Nt)+1, round(x * Nx), round(y * Ny)

m = 1
g = 0

# some changes of variables
DX = 0.1
DT = 0.1
d = 2
zeta = DX/DT
ks = DX**(d-3)*DT/2
kappa = zeta*ks
lam = g*zeta*ks*ks/6/DX**(d-4)


N_SWEEPS = 100
MONTE_CARLO_N_ROUNDS = 1
MONTE_CARLO_WORKER_COUNT = 1000
PHI_SHAPE = (MONTE_CARLO_WORKER_COUNT, Nt, Nx, Ny)
PHI_BYTES = np.product(PHI_SHAPE, dtype=np.int64)*4

# START GPU CODE =============================
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as cl_random

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

rendered_code = render(open('phi4.cl').read())
with open('phi4_rendered.cl', 'w') as f:
    f.write(rendered_code)

prg = cl.Program(ctx, rendered_code).build()

# END GPU CODE =============================

# START OBSERVATIONS =========================
phi_global = np.zeros(PHI_SHAPE, dtype=np.float32)
phi = np.zeros((Nt, Nx, Ny), dtype=np.float32)
deltas_g = cl_array.empty(queue, PHI_SHAPE, dtype=np.float32)
probs_g = cl_array.empty(queue, PHI_SHAPE, dtype=np.float32)
mu_g = cl_array.empty(queue, PHI_SHAPE, dtype=np.float32)
DELTA = 1.1

rng_g = cl_random.PhiloxGenerator(ctx)

input(f'Expected GPU memory usage {round((deltas_g.nbytes + probs_g.nbytes + mu_g.nbytes + phi_global.nbytes + 4*MONTE_CARLO_N_ROUNDS)/1000000000, 3)} GB. Press enter to start.')


for n in range(MONTE_CARLO_N_ROUNDS):
    print('Run %s...' % n)
    acceptance_rate_metro_g = cl_array.zeros(queue, MONTE_CARLO_WORKER_COUNT, dtype=np.int32)
    acceptance_rate_micro_g = cl_array.zeros(queue, MONTE_CARLO_WORKER_COUNT, dtype=np.int32)
    phi_global_g = cl_array.to_device(queue, phi_global)

    for sweep_j in range(N_SWEEPS):
        print(f'Sweep {sweep_j}...', end=' ')
        t = time.time()
        rng_g.fill_uniform(probs_g, queue=queue).wait()
        rng_g.fill_uniform(deltas_g, a=-DELTA, b=DELTA, queue=queue).wait()

        prg.metropolis_sweep(queue, (MONTE_CARLO_WORKER_COUNT,), None, deltas_g.data, probs_g.data, phi_global_g.data, acceptance_rate_metro_g.data).wait()

        if 0:
            rng_g.fill_uniform(probs_g, queue=queue).wait()
            rng_g.fill_uniform(deltas_g, a=-DELTA, b=DELTA, queue=queue).wait()
            rng_g.fill_uniform(mu_g, queue=queue).wait()

            prg.microcanonical_sweep_lambda0(queue, (MONTE_CARLO_WORKER_COUNT,), None, deltas_g.data, probs_g.data, mu_g.data, phi_global_g.data, acceptance_rate_micro_g.data).wait()

        acc_metro = np.round(np.mean(acceptance_rate_metro_g.get(queue)).astype(float) / (sweep_j+1) / Nt / Nx / Ny, 3)*100
        acc_micro = np.round(np.mean(acceptance_rate_micro_g.get(queue)).astype(float) / (sweep_j+1) / Nt / Nx / Ny, 3) * 100
        print('done in %.2fs. Metro accepted %.1f%%. Micro accepted %.1f%%.' % (time.time()-t, acc_metro, acc_micro))

    cl.enqueue_copy(queue, phi_global, phi_global_g.data)

    for p in phi_global:
        new = p*np.prod(p[:,Nx//2,Ny//2])
        if not np.any(np.logical_or(np.isnan(new), np.isinf(new))):
            phi += new
        else:
            print('bad encountered!')


print(f'Done! N={MONTE_CARLO_WORKER_COUNT*MONTE_CARLO_N_ROUNDS} observations collected')
phi /= MONTE_CARLO_WORKER_COUNT*MONTE_CARLO_N_ROUNDS
# END OBSERVATIONS =========================

# START GRAPHING ===================
np.save('phi', phi)

print('Graphing...')

fig, ax1 = plot.subplots(1,1)
image = ax1.imshow(phi[1, :, :], origin='lower', interpolation='bilinear')#, extent=[-XDIM/2, XDIM/2, -YDIM/2, YDIM/2])
plot.colorbar(image)
def animate_func(i):
    image.set_array(phi[i, :, :])
    #p.set_ydata(real[i,:])
    plot.title(r'$\langle\phi\phi\rangle$ t=%s' % i)
    return image,

anim = animation.FuncAnimation(
   fig,
   animate_func,
   frames = Nt,
   interval = 50, # in ms
)

plot.xlabel('X')
plot.ylabel('Y')
plot.show()