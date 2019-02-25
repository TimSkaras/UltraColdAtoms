# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 08:57:17 2019

@author: tskar
"""
import numpy as np
import matplotlib.pyplot as plt

"""
NOTES:
 - L should be sufficiently large that neumann boundary conditions are accurate
 - Assume points are allocated such that if L = 10 then the first point on the 
   left is at x = -5 and the last point on the right is at at x = 5
 - ITP Hamiltonian equation becomes
     d\Psi/dt = - (0.5*(- Dxx + x**2) + lam*x**4)*\Psi

"""

def Dxx(waveFunction):
    # Finds second derivative at each point assuming Neumann BC
    derivs = np.zeros(numpts)
    # Calculate derivative transformation
    for i in range(1, numpts-1):
        derivs[i] = (waveFunction[i - 1] - 2*waveFunction[i] + waveFunction[i+1])/h**2 
    
    # Take care of boundaries
    derivs[0] = (-1*waveFunction[0] + waveFunction[1])/h**2
    derivs[numpts-1] = (waveFunction[numpts-2] - 1*waveFunction[numpts-1])/h**2
    
    return derivs

def getEnergy(waveFunction):
    # Finds the energy of a wave function assuming harmonic hamiltonian
    
    energy = np.sum(-np.conj(waveFunction)*0.5*Dxx(waveFunction)+ np.conj(waveFunction)*(0.5*x**2+lam*x**4)*waveFunction)*h
    
    return energy

def getNorm(waveFunction):
    # Finds the norm of a wave function

    return np.sqrt(np.sum(np.abs(waveFunction)**2)*h)    
        
def unperturbedSol(energyLevel):
    """
    Finds the unperturbed single particle harmonic solution for wave function
    with n = energyLevel and returns array of vals over num_points
    
    INPUTS:
        energyLevel -- integer indicating what eigenstate level n =0,1,2,...
        
    OUTPUTS:
        harmSol -- numpy array with solution values
    
    """
    
    prefactor = 1./(np.pi**0.25 * np.sqrt(2**energyLevel * np.math.factorial(energyLevel)))
    
    polyDegree = np.zeros(energyLevel + 1)
    polyDegree[-1] = 1
    harmSol = np.polynomial.hermite.hermval(x, polyDegree)
    harmSol = harmSol * np.exp(-x**2/2.) * prefactor
    
    return harmSol

def findGroundState(trialSolution, tolerance):
    """
    This function finds the ground state for a spinless boson using the 
    imaginary time propagation method
    
    INPUT:
        trialSolution -- array containing values of phi with initial guess
        tolerance -- double for how close final energy should be to actual energy (e.g., 10**-3, 10**-4, etc.)
        
    OUTPUT:
        groundState -- ground state of spinless Hamiltonian
        energyPlot -- history of the energy of the wave function at each step
    """
    
    delta = 1.0
    groundState = trialSolution
    new_gs = groundState
    maxIter = 20000
    iterations = 0
    deriv = np.zeros(numpts)
    energy = getEnergy(trialSolution)
    energyPlot = np.array([energy])
    tplot = np.array([0.0])
    
    while delta > tolerance:
        
        iterations += 1
        
        # Calculate derivative transformation
        deriv = 0.5*Dxx(groundState)
        
        # Potential term
        deriv = deriv - (0.5 * x**2 + lam * x**4) * groundState
        
        new_gs = groundState + dt * deriv
        
        if iterations % 10 == 0:
            groundState = groundState/getNorm(groundState)
            energy = getEnergy(groundState)
            delta = np.abs(energy - energyPlot[-1])/energy/dt
            energyPlot = np.append(energyPlot, energy)
            tplot = np.append(tplot, dt*iterations)
            
        groundState = new_gs
        
        # Break if max iterations exceeded
        if iterations > maxIter:
            break
    
    groundState = groundState/getNorm(groundState)
    return groundState, energyPlot, tplot

def momentumSpace(waveFunction):
    """
    This function takes a wave function in real space specified over length L 
    and centered at 0. Then it calculates this wave function in momentum space
    assuming that the largest momentum will be less than p = N*pi/L
    
    OUTPUTS:
        momentumWF -- array of N values for different values of p
    """
    
#    momentumWF = np.zeros(numpts, dtype=np.complex_)
    momentumWF = np.sum(np.exp(-1j*np.transpose([p])* [x])*[waveFunction], axis=1, dtype=np.complex_)
    momentumWF = momentumWF/np.sqrt(2*np.pi)*h
    return momentumWF

def timeEvolve(momentumWF, t):
    """
    This function takes a wave function in momentum space and finds the 
    value of that wave function at time t in position space where t is 
    a vector of different times
    """
    
    waveFunction = np.zeros((numpts, len(t)), dtype=np.complex_)
    
    # x is contant along the same row and p changes
    A = np.exp(1j*np.transpose([x])*[p])
    B = np.array([momentumWF])
    C = np.exp(-1j*np.transpose([p**2])*[t]/2.)
    waveFunction = np.matmul(A*B,C)
    waveFunction = waveFunction/np.sqrt(2*np.pi)*h
    return np.transpose(waveFunction)
    
# Important variables
L = 20.0  # Trap is centered at 0 with total length L
numpts = 200
h = L/(numpts - 1)
lam = 0.1  # perturbative parameter on anharmonic term
dt = .001  
tol = 10**-8
x = np.linspace(-L/2., L/2., numpts)
p = np.linspace(-L/2., L/2., numpts)

# Important for preventing instability
print(r'Courant Condition: 1- 4 dt/dx^2 = ' + f'{1.0 - 4*dt/h**2}')

trial0 = unperturbedSol(0)
phi0, gsEnergy, tplot0 = findGroundState(trial0, tol)
#print(f'Trial Function energy: {getEnergy(trial1)}')
#print(f'Final Computed Energy: {gsEnergy[-1]}')

trial1 = unperturbedSol(1)
phi1, phi1Energy, tplot1 = findGroundState(trial1, tol)

# Plot solutions
y1 = trial0
y2 = phi0
y3 = trial1
y4 = phi1

fig, ax = plt.subplots()
ax.plot(x, y1, 'r', label=r'Unperturbed $\phi_0$')
ax.plot(x, y2, 'b', label=r'Computed $\phi_0$')
ax.set_xlabel('x')
ax.set_ylabel(r'$|\psi|$')
ax.legend(loc='upper right')
ax.grid(linestyle=':')
ax.set_title('Ground State Trial Function vs. ITP Solution $(\lambda = $' + f'{lam})')

fig3, ax3 = plt.subplots()
ax3.plot(x, y3, 'r', label=r'Unperturbed $\phi_1$')
ax3.plot(x, y4, 'b', label=r'Computed $\phi_1$')
ax3.set_xlabel('x')
ax3.set_ylabel(r'$|\psi|$')
ax3.legend(loc='upper right')
ax3.grid(linestyle=':')
ax3.set_title(r'First Excited State Trial Function vs. ITP Solution $(\lambda = $' + f'{lam})')

# Plot energy convergence
fig2, ax2 = plt.subplots()
ax2.plot(tplot0, gsEnergy-gsEnergy[-1], label=r'$\phi_0$ Energy Convergence')
ax2.plot(tplot1, phi1Energy - phi1Energy[-1], label=r'$\phi_1$ Energy Convergence')
ax2.set_title('Energy Plot $(\lambda = $' + f'{lam})')
ax2.set_xlabel('Time')
ax2.set_ylabel('Energy')
ax2.legend(loc='upper right')
ax2.grid(linestyle=':')
plt.show()

phi0_kspace = np.zeros(numpts, dtype=np.complex_)
phi1_kspace = np.zeros(numpts, dtype=np.complex_)
phi0_time = np.zeros(numpts, dtype=np.complex_)

# Now let's find the momentum distribution
phi0_kspace = np.real(momentumSpace(phi0))
phi1_kspace = np.imag(momentumSpace(phi1))

# Plot the momentum distributions
fig4, ax4 = plt.subplots()
#ax4.plot(x, y3, 'r', label=r'Unperturbed $\phi_1$')
ax4.plot(x, phi0, 'r', label=r'Unperturbed $\phi_0(p)$')
ax4.plot(p, phi0_kspace, 'b', label=r'Computed $\phi_0(p)$')
ax4.set_xlabel('p')
ax4.set_ylabel(r'$|\psi(p)|$')
ax4.legend(loc='upper right')
ax4.grid(linestyle=':')
ax4.set_title('Momentum Space G.S. Wave Function $(\lambda = $' + f'{lam})')

#%%
# Use function to find matrix with phi(x, t) at different t for phi0 & phi1
time_vector = np.array([0,1,2,3,4])
phi0_time = timeEvolve(momentumSpace(phi0), time_vector)
phi1_time = timeEvolve(momentumSpace(phi1), time_vector)


# Plot the momentum distributions
fig5, ax5 = plt.subplots()
#ax4.plot(x, y3, 'r', label=r'Unperturbed $\phi_1$')
ax5.plot(x, np.abs(phi1)**2, 'r', label=r'Unperturbed $\phi_0(p)$')
ax5.plot(p, np.abs(phi1_time[4][:])**2, 'b', label=r'Computed $\phi_0(p)$')
ax5.set_xlabel('p')
ax5.set_ylabel(r'$|\psi(p)|$')
ax5.grid(linestyle=':')
ax5.set_title('Time Evolution of G.S.')

#%% Find time evolution of OBDM

# Get slater determinant

# Integrate over second parameter for obdm


#%% Find Momentum Space Density Profile

