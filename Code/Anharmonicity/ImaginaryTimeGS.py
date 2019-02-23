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
        
        if iterations % 5 == 0:
            groundState = groundState/getNorm(groundState)
            energy = getEnergy(groundState)
            delta = np.abs(energy - energyPlot[-1])/energy/dt
            energyPlot = np.append(energyPlot, energy)
            tplot = np.append(tplot, dt*iterations)
            
        groundState = new_gs
        
        # Break if max iterations exceeded
        if iterations > maxIter:
            break
    # Calculate derivative transformation
    deriv = 0.5*Dxx(groundState)
    
    # Potential term
    deriv = deriv - (0.5 * x**2 + lam * x**4) * groundState
        
#    fig2, ax2 = plt.subplots()
#    ax2.plot(x, deriv)
    
    groundState = groundState/getNorm(groundState)
    return groundState, energyPlot, tplot
    
# Important variables
L = 15  # Trap is centered at 0 with total length L
numpts = 160
h = L/(numpts - 1)
lam = 0.1  # perturbative parameter on anharmonic term
dt = .001
tol = 10**-8
x = np.linspace(-L/2., L/2., numpts)

#trial1 = np.sin(4*2*np.pi/L*x)**2/100.
trial1 = unperturbedSol(0)
print(np.sum(trial1*lam*x**4*trial1)*h) # first order correction to energy
print(r'Courant Condition: 1- 4 dt/dx^2 = ' + f'{1.0 - 4*dt/h**2}')
gs1, gsEnergy, tplot = findGroundState(trial1, tol)
print(f'Trial Function energy: {getEnergy(trial1)}')
print(f'Final Computed Energy: {gsEnergy[-1]}')

# Construct analytic perturbed
analyticPerturbation = -(3./(2.*np.sqrt(2.)) * unperturbedSol(2) + (np.sqrt(3./2)/4.)*unperturbedSol(4))
firstOrderSol = trial1 + lam*analyticPerturbation
# Plot solutions
y1 = trial1
y2 = gs1
y3 = firstOrderSol
#print(getNorm(y))
fig, ax = plt.subplots()
ax.plot(x, y1, 'r', label='Unperturbed')
ax.plot(x, y2, 'b', label='Perturbed')
ax.plot(x, y3, 'g', label='Perturbed Analytic')
ax.set_xlabel('x')
ax.set_ylabel(r'$|\Psi|^2$')
ax.legend(loc='upper right')
ax.grid(linestyle=':')
ax.set_title('Trial Function vs. ITP Solution')

# Plot energy convergence
fig2, ax2 = plt.subplots()
ax2.plot(tplot, gsEnergy - gsEnergy[-1])
ax2.set_title('Energy Plot')
ax2.set_xlabel('Time')
ax2.set_ylabel('Energy')
ax2.grid(linestyle=':')
plt.show()