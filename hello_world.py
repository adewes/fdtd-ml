from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from matplotlib.pylab import *

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

    """


    mod = SourceModule("""

    #include <stdio.h>

    __device__ int mod(int a, int b)
    {
        int r = a % b;
        return r < 0 ? r + b : r;
    }

    __global__ void fdtd(
        int nx,
        int ny,
        float *alpha,//time-independent
        float *beta,//time-independent
        float *Ez,
        float *Hx,
        float *Hy,
        float *Jz,
        float *Ezp,//new value of Ez
        float *Hxp,//new value of Hx
        float *Hyp //new value of Hy
        )
        {
        int i = threadIdx.x + blockDim.x*blockIdx.x;
        int j = threadIdx.y + blockDim.y*blockIdx.y;
        int idx = i*ny+j;

        if (j + i == ny /2 ){
            Jz[idx] = 1;
        }

        int im = mod(i-1, nx);
        int ip = mod(i+1, nx);
        int jm = mod(j-1, ny);
        int jp = mod(j+1, ny);

        //printf("%d, %d\\n", i, j);

        const float dt = 0.000001;
        const float dx = 0.001;
        const float dy = 0.001;
        const float mu = 100.0;

        //H_x^{n+1}(i,j) = H_x^{n}(i,j)
        //    -\Delta t /(\mu \Delta y)*(E_z^n(i,j+1)-E_z^n(i,j))

        Hxp[idx] = Hx[idx] - dt/(mu*dy)*(Ez[i*ny+jp]-Ez[idx]);

        //H_y^{n+1}(i,j) = H_y^{n}(i,j)
        //    +\Delta t /(\mu \Delta x)*(E_z^n(i+1,j)-E_z^n(i,j))

        Hyp[idx] = Hy[idx] + dt/(mu*dy)*(Ez[ip*ny+j]-Ez[idx]);

        //E_z^{n+1}(i,j) = 1/\beta(i,j)*(
        //    \alpha(i,j)*E_z^n(i,j)
        //    +1/\Delta x*(H_y^{n+1}(i,j) - H_y^{n+1}(i-1,j) )
        //    -1/\Delta y*(H_x^{n+1}(i,j) -H_x^{n+1}(i,j-1))-J_z^{n+1}(i,j)
        //)

        Ezp[idx] = 1/beta[idx]*(
            alpha[idx]*Ez[idx] +
            (1.0/dx)*(Hyp[idx]-Hyp[im*ny+j]) -
            (1.0/dy)*(Hxp[idx]-Hxp[i*ny+jm]) -
            Jz[idx]
        );
        
    }
    """)


    def __init__(self, nx=256, ny=256):

        self.fdtd = self.mod.get_function("fdtd")

        self.nx = np.int32(nx)
        self.ny = np.int32(ny)

        self.block_x = 16
        self.block_y = 16

        self.threads = self.block_x*self.block_y

        #we need to cover nx*ny elements
        self.grid_x = math.ceil(nx/self.block_x)
        self.grid_y = math.ceil(ny/self.block_y)

        def to_gpu(A):
            return gpuarray.to_gpu(A)

        self.dt = np.float32

        self.Hx = to_gpu(np.zeros((nx, ny), dtype=self.dt))
        self.Hy = to_gpu(np.zeros((nx, ny), dtype=self.dt))
        self.Ez = to_gpu(np.zeros((nx, ny), dtype=self.dt))

        self.Hxp = to_gpu(np.zeros((nx, ny), dtype=self.dt))
        self.Hyp = to_gpu(np.zeros((nx, ny), dtype=self.dt))
        self.Ezp = to_gpu(np.zeros((nx, ny), dtype=self.dt))
        self.Jz = to_gpu(np.zeros((nx, ny), dtype=self.dt))

        self.alpha = to_gpu(np.zeros((nx, ny), dtype=self.dt)+1.91)
        self.beta = to_gpu(np.zeros((nx, ny), dtype=self.dt)+1.91)

    def evolve(self):
        self.fdtd(
            self.nx,
            self.ny,
            self.alpha,
            self.beta,
            self.Ez,
            self.Hx,
            self.Hy,
            self.Jz,
            self.Ezp,
            self.Hxp,
            self.Hyp,
            block=(self.block_x,self.block_y,1),
            grid=(self.grid_x, self.grid_y, 1)
        )
        self.Ezp, self.Ez = self.Ez, self.Ezp
        self.Hxp, self.Hx = self.Hx, self.Hxp
        self.Hyp, self.Hy = self.Hy, self.Hyp

if __name__ == '__main__':


    simulation = Simulation()

    for i in range(1000):
        simulation.evolve()
    figure(figsize=(20,10))
    subplot(1,3,1)
    imshow(np.array(simulation.Hx.get()))
    subplot(1,3,2)
    imshow(np.array(simulation.Hy.get()))
    subplot(1,3,3)
    imshow(np.array(simulation.Ez.get()))
    show()
