from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
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

        //printf("%d, %d\\n", i, j);

        const float dx = 0.01;
        const float dy = 0.01;

        Hxp[idx] = 1/betay[idx]*(
            alphay[idx]*Hx[idx] - epsilon[idx]/(mu[idx]*dy)*(Exz[i*ny+jp]-Exz[idx]+Eyz[i*ny+jp]-Eyz[idx])
        );

        Hyp[idx] = 1/betax[idx]*(
            alphax[idx]*Hy[idx] + epsilon[idx]/(mu[idx]*dy)*(Exz[ip*ny+j]-Exz[idx]+Eyz[ip*ny+j]-Eyz[idx])
        );

        /*
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
        */

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
        
    }
    """)


    def __init__(self, nx=256, ny=256, s=1.0, p=(230, 120), sp=(180,210,120,150)):

        self.fdtd = self.mod.get_function("fdtd")

        self.dtype = np.float32

        self.p = p
        self.sp = sp

        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.t = self.dtype(0.0)
        self.dt = self.dtype(0.001)

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

        shape[self.sp[0]:self.sp[1],self.sp[2]:self.sp[3]] = s

        epsilon+=shape*0.6

        sig = 100.0

        ax = np.zeros((nx, ny), dtype=self.dtype)+epsilon/self.dt
        ay = np.zeros((nx, ny), dtype=self.dtype)+epsilon/self.dt
        bx = np.zeros((nx, ny), dtype=self.dtype)+epsilon/self.dt
        by = np.zeros((nx, ny), dtype=self.dtype)+epsilon/self.dt

        ax -= shape*sig+0.04
        ay -= shape*sig+0.04

        bx += shape*sig+0.04
        bx += shape*sig+0.04

        l = 20  
        for i in range(l):

            v = (l-i)*8.0/float(l)

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
        if last_t == 0:
            jzd = np.zeros((self.nx, self.ny), dtype=self.dtype)
            jzd[self.p[0],self.p[1]] = 100.0
            self.Jzd = to_gpu(jzd)
        elif last_t < 1.0 and self.t > 1.0:
            jzd = np.zeros((self.nx, self.ny), dtype=self.dtype)
            self.Jzd = to_gpu(jzd)

        self.fdtd(
            self.t,
            self.dt,
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

def run_simulation(simulation, verbose=False):
    results = []
#    fig = figure(figsize=(20,20))
#    fig.show()
    for j in range(1000):
        if verbose and j % 100 == 99:
            print(simulation.t)
        for i in range(10):
            simulation.evolve()
        eza = np.array(simulation.Exz.get()+simulation.Eyz.get())
        results.append(eza)
#        plot_sim(simulation, fig)
    return results

def plot_sim(simulation, fig):
    eza = np.array(simulation.Exz.get()+simulation.Eyz.get())
    lim = list((np.min(eza), np.max(eza)))
    clf()
    winter()
    subplot(1,3,1)
    imshow(np.array(simulation.Hx.get()))
    subplot(1,3,2)
    imshow(np.array(simulation.Hy.get()))
    subplot(1,3,3)
    imshow(eza)
    clim(*lim)
    fig.canvas.draw()
    print("Done!")

if __name__ == '__main__':
    ps = ((230,40),(230,80),(230,120),(230,160),(230,200))
    results_bl = []
    print("Calculating baselines...")
    for p in ps:
        simulation_bl = Simulation(256, 256, 0.0, p=p)
        results_bl.append(run_simulation(simulation_bl))
    print("Done calculating baselines...")

    for i in range(10):
        line = []

        print("Calculating signals...")
        x = random.randint(40,180)
        y = random.randint(40,180)
        sp=(y,y+30,x,x+30)
        for j, p in enumerate(ps):
            simulation = Simulation(256, 256, 1.0, p=p, sp=sp)
            results = run_simulation(simulation)
            for i in range(len(results)):
                r = results[i]
                rb = results_bl[j][i]
                line.append(r[230,:]-rb[230,:])
        #        line.append(r[230,:])
        line = np.array(line)
        imshow(line, aspect='auto')
        show()
        plot(line[:,40])
        plot(line[:,80])
        plot(line[:,120])
        plot(line[:,160])
        plot(line[:,200])
        show()