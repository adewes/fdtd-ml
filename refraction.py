from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from PIL import Image
import pycuda.autoinit
import numpy as np
import time

from matplotlib.pylab import *
from PyQt5 import QtWidgets

import random

def to_gpu(A):
    return gpuarray.to_gpu(A)

class Simulation():

    """
    2D FDTD Simulation:

    H_x^{n+1/2}(i,j+1/2) = H_x^{n-1/2}(i,j+1/2)
        -\Delta t /(\mu \Delta y)*(E_z^n(i,j+1)-E_z^n(i,j))

    H_y^{n+1/2}(i+1/2,j) = H_y^{n-1/2}(i+1/2,j)
        +\Delta t /(\mu \Delta x)*(E_z^n(i+1,j)-E_z^n(i,j))

    E_z^{n+1}(i,j) = 1/\beta(i,j)*(
        \alpha(i,j)*E_z^n(i,j)
        +1/\Delta x*(H_y^{n+1/2}(i+1/2,j) - H_y^{n+1/2}(i-1/2,j) )
        -1/\Delta y*(H_x^{n+1/2}(i,j+1/2) -H_x^{n+1/2}(i,j-1/2))-J_z^{n+1/2}(i,j)
    )

    \alpha = \epsilon / \Delta t - \sigma / 2
    \beta = \epsilon / \Delta t + \sigma / 2

    alpha, beta and E_z are defined on the grid (i,j)

    H_x and H_y are defined on the grid

    To add a simple PML boundary condition, we need to modify the equations
    as follows:

    H_x^{n+1/2}(i,j+1/2) = 1/\beta_y(i,j+1/2)*(
        \alpha_y(i,j+1/2)*H_x^{n-1/2}(i,j+1/2)
        -\epsilon/(\mu\Delta y)*(E_z^n(i,j+1)-E_z^n(i,j))
    )

    H_y^{n+1/2}(i+1/2,j) = 1/\beta(i+1/2,j)*(
        H_y^{n-1/2}(i+1/2,j)\alpha_x(i+1/2,j)
        +\epsilon/(\mu\Delta x)*(E_z^n(i+1,j)-E_z^n(i,j))
    )

    E_{sx,z}^{n+1}(i,j) = 1/\beta_x(i,j)*(
        E_{sx,z}^n(i,j)\alpha_x(i,j)+
        1/\Delta x*(
            H_y^{n+1/2}(i+1/2,j)-H_y^{n+1/2}(i-1/2,j)
        )
    )

    E_{sy,z}^{n+1}(i,j) = 1/\beta_y(i,j)*(
        E_{sy,z}^n(i,j)*\alpha_y(i,j)
        -1/\Delta y*(
            H_x^{n+1/2}(i,j+1/2)-H_x^{n+1/2}(i,j-1/2)
        )
    )

    E_z = E_{sx,z} + E_{sy,z}

    \alpha_{x,y} = \epsilon/Delta t - \sigma_{x,y}/2
    \beta_{x,y} = \epsilon/Delta t + \sigma_{x,y}/2

    """


    mod = SourceModule("""

    #include <stdio.h>

    __device__ int mod(int a, int b)
    {
        int r = a % b;
        return r < 0 ? r + b : r;
    }

    __global__ void fdtd(
        float t,
        float dt,
        float dx,
        float dy,
        int nx,
        int ny,
        float *alphax,//time-independent
        float *betax,//time-independent
        float *alphay,//time-independent
        float *betay,//time-independent
        float *epsilon,
        float *mu,
        float *Exz,
        float *Eyz,
        float *Hx,
        float *Hy,
        float *Jz,
        float *Jzd,
        float *Exzp,//new value of Ez
        float *Eyzp,//new value of Ez
        float *Hxp,//new value of Hx
        float *Hyp //new value of Hy
        )
        {
        int i = threadIdx.x + blockDim.x*blockIdx.x;
        int j = threadIdx.y + blockDim.y*blockIdx.y;
        int idx = i*ny+j;

        if (Jzd[idx] != 0){
            Jz[idx] = sin(t*30.0)*Jzd[idx];
        }

        int im = mod(i-1, nx);
        int ip = mod(i+1, nx);
        int jm = mod(j-1, ny);
        int jp = mod(j+1, ny);

        Hxp[idx] = 1/betay[idx]*(
            alphay[idx]*Hx[idx] - epsilon[idx]/(mu[idx]*dy)*(Exz[i*ny+jp]-Exz[idx]+Eyz[i*ny+jp]-Eyz[idx])
        );

        Hyp[idx] = 1/betax[idx]*(
            alphax[idx]*Hy[idx] + epsilon[idx]/(mu[idx]*dy)*(Exz[ip*ny+j]-Exz[idx]+Eyz[ip*ny+j]-Eyz[idx])
        );

        Exzp[idx] = 1/betax[idx]*(
            alphax[idx]*Exz[idx] +
            (1.0/dx)*(Hyp[idx]-Hyp[im*ny+j]) -
            Jz[idx]/2.0
        );

        Eyzp[idx] = 1/betay[idx]*(
            alphay[idx]*Eyz[idx] -
            (1.0/dy)*(Hxp[idx]-Hxp[i*ny+jm]) -
            Jz[idx]/2.0
        );

        return;        
    }
    """)


    def __init__(self, nx=256, ny=256, p=(230, 120)):

        self.fdtd = self.mod.get_function("fdtd")
        self.fdtd.prepare("P")

        self.dtype = np.float32

        self.p = p

        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.t = self.dtype(0.0)
        self.dt = self.dtype(0.01)
        self.dx = self.dtype(0.01)
        self.dy = self.dtype(0.01)

        self.block_x = 16
        self.block_y = 16

        self.threads = self.block_x*self.block_y

        #we need to cover nx*ny elements
        self.grid_x = math.ceil(nx/self.block_x)
        self.grid_y = math.ceil(ny/self.block_y)

        self.Hx = to_gpu(np.zeros((nx, ny), dtype=self.dtype))
        self.Hy = to_gpu(np.zeros((nx, ny), dtype=self.dtype))
        self.Exz = to_gpu(np.zeros((nx, ny), dtype=self.dtype))
        self.Eyz = to_gpu(np.zeros((nx, ny), dtype=self.dtype))

        self.Hxp = to_gpu(np.zeros((nx, ny), dtype=self.dtype))
        self.Hyp = to_gpu(np.zeros((nx, ny), dtype=self.dtype))
        self.Exzp = to_gpu(np.zeros((nx, ny), dtype=self.dtype))
        self.Eyzp = to_gpu(np.zeros((nx, ny), dtype=self.dtype))

        self.Jz = to_gpu(np.zeros((nx, ny), dtype=self.dtype))
        self.Jzd = to_gpu(np.zeros((nx, ny), dtype=self.dtype))

        # \alpha_{x,y} = \epsilon/Delta t - \sigma_{x,y}/2
        # \beta_{x,y} = \epsilon/Delta t + \sigma_{x,y}/2

        epsilon = np.zeros((nx, ny), dtype=self.dtype)+1.0
        mu = np.zeros((nx, ny), dtype=self.dtype)+2.0*3.14

        shape = np.zeros((nx, ny), dtype=self.dtype)

        #Parabolic antenna:
        #for x in range(nx):
        #    if x < nx//4 or x > nx*3//4:
        #        continue
        #    ax = int(float((x-nx//2)**2)/float(nx**2//4)*nx*2//7)
        #    for h in range(20):
        #        shape[nx//4+ax-h, x] = 1.0

        r = nx//12
        x = nx//2
        y = ny//2

        for xx in range(2*r):
            for yy in range(2*r):
                dd = ((xx-r)**2+(yy-r)**2)/r**2
                if dd <= 1:
                    shape[x+xx-r,y+yy-r] = 1.0-np.exp(-(1.0-dd)*40.0)

        self.shape = shape

        epsilon+=shape*0.77

        sig = 0.0

        ax = np.zeros((nx, ny), dtype=self.dtype)+epsilon/self.dt
        ay = np.zeros((nx, ny), dtype=self.dtype)+epsilon/self.dt
        bx = np.zeros((nx, ny), dtype=self.dtype)+epsilon/self.dt
        by = np.zeros((nx, ny), dtype=self.dtype)+epsilon/self.dt

        loss = 0.0

        ax -= shape*sig+loss
        ay -= shape*sig+loss

        bx += shape*sig+loss
        bx += shape*sig+loss

        l = 20  
        for i in range(l):

            v = (l-i)*16.0/float(l)

            ax[i,:] -= v
            bx[i,:] += v

            ax[-i-1,:] -= v
            bx[-i-1,:] += v

            ay[:,i] -= v
            by[:,i] += v

            ay[:,-i-1] -= v
            by[:,-i-1] += v

        self.epsilon = to_gpu(epsilon)
        self.mu = to_gpu(mu)

        self.alphax = to_gpu(ax)
        self.alphay = to_gpu(ay)

        self.betax = to_gpu(bx)
        self.betay = to_gpu(by)

    def evolve(self):
        last_t = self.t
        self.t += self.dt

        j = int(self.t//1)
        last_j = int(last_t//1)
        jzd = np.zeros((self.nx, self.ny), dtype=self.dtype)
        for x in range(self.nx-40):
            jzd[self.p[0]+x-self.nx//2+20,self.p[1]] = 5.0*np.exp(-(self.t-3.0)**2*3.0)
        self.Jzd = to_gpu(jzd)

        self.fdtd(
            self.t,
            self.dt,
            self.dx,
            self.dy,
            self.nx,
            self.ny,
            self.alphax,
            self.betax,
            self.alphay,
            self.betay,
            self.epsilon,
            self.mu,
            self.Exz,
            self.Eyz,
            self.Hx,
            self.Hy,
            self.Jz,
            self.Jzd,
            self.Exzp,
            self.Eyzp,
            self.Hxp,
            self.Hyp,
            block=(self.block_x,self.block_y,1),
            grid=(self.grid_x, self.grid_y, 1)
        )
        self.Exzp, self.Exz = self.Exz, self.Exzp
        self.Eyzp, self.Eyz = self.Eyz, self.Eyzp
        self.Hxp, self.Hx = self.Hx, self.Hxp
        self.Hyp, self.Hy = self.Hy, self.Hyp

def to_jet(a, rgb):
    ev = a*4.0
    rgb[:,:,0] = (np.clip(np.minimum(ev-1.5, -ev+4.5), 0, 1.0)*255).astype(np.uint8)
    rgb[:,:,1] = (np.clip(np.minimum(ev-0.5, -ev+3.5), 0, 1.0)*255).astype(np.uint8)
    rgb[:,:,2] = (np.clip(np.minimum(ev+0.5, -ev+2.5), 0, 1.0)*255).astype(np.uint8)
    return rgb

def mix_rgb(a, b, ratio):
    c = np.zeros(a.shape, dtype=np.uint8)
    for i in range(3):
        c[:,:,i] = a[:,:,i]*ratio+b[:,:,i]*(1.0-ratio-0.01)
    return c
    
def run_simulation(simulation, verbose=True, save=True):
    fig = figure(figsize=(12,12), frameon=False)
    fig.show()
    st = time.time()
    N = 5000
    s = (simulation.nx, simulation.ny, 3)
    srgb = np.zeros(s, dtype=np.uint8)
    rgb = np.zeros(s, dtype=np.uint8)
    to_jet(simulation.shape, srgb)
    for j in range(N):
        if verbose and j % 50 == 49:
            print(j)
            plot_simulation(simulation, fig)
        simulation.evolve()
        if save:
            eza = np.array(simulation.Exz.get()+simulation.Eyz.get())
            ev = np.clip(((eza+0.06)/0.12), 0, 1)
            to_jet(ev, rgb)
            img = Image.fromarray(mix_rgb(srgb, rgb, 0.2), 'RGB')
            img = img.resize((rgb.shape[0]*2, rgb.shape[1]*2), Image.BILINEAR)
            img.save('img-{:04d}.png'.format(j))

    print("Frames/second: {:.4f}".format(1.0/(time.time()-st)*float(N)))
    return results

def plot_simulation(simulation, fig):
    eza = np.array(simulation.Exz.get()+simulation.Eyz.get())
    cla()
    limits = (-0.06, 0.06)
    jet()
    imshow(simulation.shape, alpha=1.0, interpolation='nearest')
    imshow(eza, alpha=0.8, interpolation='nearest')
    clim(*limits)
    ax = subplot()
    ax.axis('off')    
    fig.canvas.draw()
    #savefig("im-{:03d}.png".format(int(simulation.t*100)))

if __name__ == '__main__':
    simulation = Simulation(512, 512, p=(255, 490))
    results = run_simulation(simulation)
