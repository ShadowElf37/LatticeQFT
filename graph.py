import matplotlib.pyplot as plot
import matplotlib.animation as animation
import numpy as np

phi = np.load('phi.npy')
Nt = phi.shape[0]

fig, ax1 = plot.subplots(1,1)
image = ax1.imshow(phi[0, :, :], origin='lower')#, interpolation='bilinear')#, extent=[-XDIM/2, XDIM/2, -YDIM/2, YDIM/2])
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