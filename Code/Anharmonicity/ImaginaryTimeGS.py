# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 08:57:17 2019

@author: tskar
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import diags
import sys
from matplotlib import animation

"""
TODO:
    - Output OBDM does not have correct norm

NOTES:
 - L should be sufficiently large that neumann boundary conditions are accurate
 - Assume points are allocated such that if L = 10 then the first point on the 
   left is at x = -5 and the last point on the right is at at x = 5
 - ITP Hamiltonian equation becomes
     d\Psi/dt = - (0.5*(- Dxx + x**2) + alpha*x**3 + lam*x**4)*\Psi

"""
#%%
def innerProduct(func1, func2):
    """
    This function takes two functions and finds their L2 inner product over length L
    
    INPUT:
    func1, func2 -- two functions represented as discrete sets of points over L
    
    OUTPUT:
    inner -- value of inner product
    """
    
    inner = np.sum(func1*func2) * h
    
    return inner
    

def Dxx(waveFunction):
    # Finds second derivative at each point assuming Neumann BC
    numpts = len(waveFunction)
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
    wm = waveFunction[0:numpts-1]
    wp = waveFunction[1:numpts]
    x = -L/2. + h*np.arange(numpts-1)
    WF_eff = 0.5*(wm + wp)
    
    energy = np.sum(-np.conj(WF_eff)*0.5*Dxx(WF_eff)+ np.conj(WF_eff)*(0.5*x**2+alpha*x**3+lam*x**4)*WF_eff)*h
    
    return energy

def getNorm(waveFunction):
    # Finds the norm of a wave function
    wm = waveFunction[0:numpts-1]
    wp = waveFunction[1:numpts]
    
    WF_eff = 0.5*(wm + wp)
    
    return np.sqrt(np.sum(np.abs(WF_eff)**2)*h)    
        
def unperturbedSol(energyLevel, x):
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

def findGroundState(trialSolution, tolerance, orthogonalize):
    """
    This function finds the ground state for a spinless boson using the 
    imaginary time propagation method
    
    INPUT:
        trialSolution -- array containing values of phi with initial guess
        tolerance -- double for how close final energy should be to actual energy (e.g., 10**-3, 10**-4, etc.)
        orthogonalize -- boolean (0 or 1) for whether the trial functio needs to be projected orthogonally to the g.s.
        
    OUTPUT:
        groundState -- ground state of spinless Hamiltonian
        energyPlot -- history of the energy of the wave function at each step
    """
    
    delta = 1.0
    groundState = trialSolution
    new_gs = groundState
    maxIter = 50000
    iterations = 0
    deriv = np.zeros(numpts)
    energy = getEnergy(trialSolution)
    energyPlot = np.array([energy])
    tplot = np.array([0.0])
    
    r = np.float64(dt)/h**2
    diagOff = np.ones(numpts-1)*(-r/2.)
    diagCen = np.ones(numpts -2)*(1.+r)
    diagCen = np.append(r/2. + 1, diagCen)
    diagCen = np.append(diagCen, r/2. + 1)
    diagCen += (0.5*x**2 + alpha*x**3 + lam*x**4)*dt
    eqnMatrix = diags([diagOff, diagCen, diagOff], [-1,0,1]).toarray()
    eqnInv = np.linalg.inv(eqnMatrix)
    
    animationData = [groundState]
    
    while delta > tolerance:
        
        iterations += 1
        
        if method == 0:
            # Calculate derivative transformation
            deriv = 0.5*Dxx(groundState)
            
            # Potential term
            deriv = deriv - (0.5 * x**2 + alpha * x**3 + lam * x**4) * groundState
            
            new_gs = groundState + dt * deriv
        else:
            new_gs = np.einsum('ij, j', eqnInv, groundState)
        
        if iterations % 10 == 0:
            new_gs = new_gs/getNorm(new_gs)
            energy = getEnergy(new_gs)
            delta = np.abs(energy - energyPlot[-1])/energy/dt
            energyPlot = np.append(energyPlot, energy)
            tplot = np.append(tplot, dt*iterations)
            
            if alpha > 0.0 and np.min(trialSolution) < -1./10:
                new_gs = new_gs - innerProduct(new_gs, phi0) * phi0 

            
        groundState = new_gs
        animationData = np.append(animationData, [groundState], axis=0)
        
        # Break if max iterations exceeded
        if iterations > maxIter:
            break

    
    groundState = groundState/getNorm(groundState)
    return groundState, energyPlot, tplot, animationData


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
    
    OUTPUT:
        waveFunction[time index][x index]
    """
    
    waveFunction = np.zeros((numpts, len(t)), dtype=np.complex_)
    
    # x is contant along the same row and p changes
    A = np.exp(1j*np.transpose([x])*[p])
    B = np.array([momentumWF])
    C = np.exp(-1j*np.transpose([p**2])*[t]/2.)
    waveFunction = np.matmul(A*B,C)
    waveFunction = waveFunction/np.sqrt(2*np.pi)*hp
    return np.transpose(waveFunction)

def getSlater(wf1, wf2):
    """
    Takes two wave functions with each time slice of the w.f. on a single row
    and constructs the slater determinant from them
    
    OUTPUT:
        slater[time index][x_1 index][x_2 index]
    """
    A = np.transpose([wf1[0][:]])*[wf2[0][:]]
    slater = [A - np.transpose(A)]
    
    # Loop over each time slice
    for i in range(1, len(wf1)):
        A = np.transpose([wf1[i][:]])*[wf2[i][:]]
        slater = np.append(slater, [A - np.transpose(A)], axis=0)
    
    slater = slater/ np.sqrt(2)
    return slater

def getOBDM(fullWaveFunction):
    """
    Output one-body density matrix given full two body wave function
    
    OUTPUT:
        obdm[time index][x index][x prime index]
    """
    
    obdm = np.zeros(fullWaveFunction.shape, dtype=np.complex_)
    fullConj = np.conjugate(fullWaveFunction)
    
    for j in range(len(fullWaveFunction)):
        for k in range(numpts):
            for l in range(numpts):
                obdm[j, k, l] = np.inner(fullConj[j, k, :], fullWaveFunction[j, l, :])
    
#    for j in range(len(fullWaveFunction)):
#        obdm[j] = np.einsum('ik,lk', fullConj[j], fullWaveFunction[j])
            
    return obdm*h

def timeEvolveAnalytic(n, t):
    """
    Gives analytic solution to the time evolution of the single particle states
    for the analytically solveable harmonic solutions at a given energy level
    n, position x, and time t
    
    OUTPUT:
        analSol[time index][x index]
    """
    
    b = lambda t: np.sqrt(1 + t**2)
    bprime = lambda t: t/np.sqrt(1 + t**2)
    tau = lambda t : np.arctan(t)
    
    # Use unperturbed solution to get solution at t=0
    analSol = unperturbedSol(n, x/b(t))/np.sqrt(b(t))
    analSol = analSol*np.exp(1j*(x**2*bprime(t)/b(t)/2.0 - (n + 0.5)*tau(t)))
    
    return analSol

def getMSDP(obdm):
    """
    This function takes the time dependent obdm matrix and finds the time evolution
    of the momentum space density profile (MSDP)
    
    OUTPUTS:
        msdp[time idx][p idx]
    """
    msdp = np.zeros((len(obdm), numpts), dtype=np.complex_)
    
    # to evaluate this integral we need to create the cos matrix for
    # every value of p
#    cosMat = np.cos(np.einsum('i, jk', p, np.transpose([x]) - [x]))
#    cosTerm = np.cos(np.einsum('i,j',p, x))
#    sinTerm = np.sin(np.einsum('i,j',p, x))
#    msdp = np.einsum('ijk,lj,lk->il', np.real(obdm), cosTerm, cosTerm)
#    msdp += np.einsum('ijk,lj,lk->il', np.real(obdm), sinTerm, sinTerm)
    
    expTerm = np.exp(1j*np.einsum('i,j', p, x))
    msdp = np.einsum('lj, lk, ijk->il', expTerm, np.conj(expTerm), obdm)
    
    # MSDP is guaranteed to be real
    msdp = np.real(msdp)
    
    return msdp * h**2 / np.pi
    
#%%
# Important variables
method = 1 # Forward Euler = 0, Backward Euler = 1
L = 100.0  # Trap is centered at 0 with total length (old=100)
numpts = 1200
h = np.float64(L/(numpts-2)) # This is because we have neumann boundary conditions
alpha = 0.1
lam = 0.07  # perturbative parameter on anharmonic term
dt = .001   
tol = 10**-9
Lp = L
hp = h
x = -L/2. + h*(np.arange(numpts) - 1./2)
p = -L/2. + h*(np.arange(numpts) - 1./2)
# Determine start and stop index for plotting purposes
# because we don't want the whole domain when we plot
start_idx = 0
end_idx = numpts
plot_length = 15.0
if L > plot_length:
    start_idx = np.argmin(np.abs(x+plot_length/2.))
    end_idx = np.argmin(np.abs(x-plot_length/2.))

# Important for preventing instability
print(r'Courant Condition:  dt/dx^2 = ' + f'{dt/h**2}')

trial0 = unperturbedSol(0, x)
phi0, gsEnergy, tplot0, animationData = findGroundState(trial0, tol, 0)
print(f'Trial Function energy: {getEnergy(trial0)}')
print(f'Final Computed Energy: {gsEnergy[-1]}')

# Use Graham-Schmidt to make sure second trial function is orthogonal to first eigenstate
trial1 = unperturbedSol(1, x)
trial1 = trial1 - innerProduct(phi0, trial1) * phi0 / innerProduct(phi0, phi0)
phi1, phi1Energy, tplot1, animationData = findGroundState(trial1, tol, 1)

# Plot solutions
y1 = trial0
y2 = phi0
y3 = trial1
y4 = phi1

#start_idx = 0
#end_idx = numpts

fig, ax = plt.subplots()
ax.plot(x[start_idx:end_idx], y1[start_idx:end_idx], 'r', label=r'Unperturbed $\phi_0$')
ax.plot(x[start_idx:end_idx], y2[start_idx:end_idx], 'b', label=r'Computed $\phi_0$')
ax.set_xlabel('x')
ax.set_ylabel(r'$|\psi|$')
ax.legend(loc='upper right')
ax.grid(linestyle=':')
ax.set_title('Ground State Trial Function vs. ITP Solution $(\lambda = $' + f'{lam})')

## turns on interactive
#plt.ion()
#
## animation
#plt.figure(2); plt.clf();
## now plot the motion of the pendulum, assume L=1
#nplots = 100 # number of points to plot
#
#deltai = int(round((len(animationData)-1)/nplots))
#
#for i in range(0,len(animationData), 3):
#    
#    plt.clf()
#    plt.plot(x[start_idx:end_idx], y1[start_idx:end_idx], 'r', label=r'Unperturbed $\phi_0$')
#    plt.plot(x[start_idx:end_idx], animationData[i, start_idx:end_idx], 'b', label=r'Computed $\phi_0$')
#    plt.xlabel('x')
#    plt.ylabel(r'$|\psi|$')
#    plt.legend(loc='upper right')
#    plt.grid(linestyle=':')
#    plt.title('Ground State Trial Function vs. ITP Solution $(\lambda = $' + f'{lam}, frame = {i})')
#    
#    plt.draw()
#    plt.pause(.03)
## puts in a prompt to stop the program
#plt.ioff()

fig3, ax3 = plt.subplots()
ax3.plot(x[start_idx:end_idx], y3[start_idx:end_idx], 'r', label=r'Unperturbed $\phi_1$')
ax3.plot(x[start_idx:end_idx], y4[start_idx:end_idx], 'b', label=r'Computed $\phi_1$')
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

#%%

phi0_kspace = np.zeros(numpts, dtype=np.complex_)
phi1_kspace = np.zeros(numpts, dtype=np.complex_)
phi0_time = np.zeros(numpts, dtype=np.complex_)

# Now let's find the momentum distribution
phi0_kspace = np.abs(momentumSpace(phi0))
phi1_kspace = np.real(momentumSpace(phi1))

# Plot the momentum distributions
fig4, ax4 = plt.subplots()
#ax4.plot(x, y3, 'r', label=r'Unperturbed $\phi_1$')
ax4.plot(x[start_idx:end_idx], trial0[start_idx:end_idx], 'r', label=r'Unperturbed $\phi_0(p)$')
ax4.plot(p[start_idx:end_idx], phi0_kspace[start_idx:end_idx], 'b', label=r'Computed $\phi_0(p)$')
ax4.set_xlabel('p')
ax4.set_ylabel(r'$|\psi(p)|$')
ax4.legend(loc='upper right')
ax4.grid(linestyle=':')
ax4.set_title('Momentum Space G.S. Wave Function $(\lambda = $' + f'{lam})')
sys.exit()
#%%
# Use function to find matrix with phi(x, t) at different t for phi0 & phi1
time_vector = np.array([0,3,6])
phi0_time = timeEvolve(momentumSpace(phi0), time_vector) # phi0_time[time idx][x1 idx]
phi1_time = timeEvolve(momentumSpace(phi1), time_vector)

#%%
compareidx = -1

start_idx_temp = np.argmin(np.abs(x+40./2.))
end_idx_temp = np.argmin(np.abs(x-40./2.))

# Plot the momentum distributions
fig5, ax5 = plt.subplots()
#ax4.plot(x, y3, 'r', label=r'Unperturbed $\phi_1$')
ax5.plot(x[start_idx_temp:end_idx_temp], np.abs(phi0[start_idx_temp:end_idx_temp])**2, 'r', label=r'Initial $\phi_0(x)$')
ax5.plot(p[start_idx_temp:end_idx_temp], np.abs(phi0_time[compareidx][start_idx_temp:end_idx_temp])**2, 'b', label=r'Final $\phi_0(x)$')
ax5.set_xlabel('x')
ax5.set_ylabel(r'$|\psi(x)|$')
ax5.grid(linestyle=':')
ax5.legend(loc='upper right')
ax5.set_title('Time Evolution of G.S.')
#
## Check to make sure computational answer matches theoretical answer
#diff = np.abs(phi0_time[compareidx][:] - timeEvolveAnalytic(0, time_vector[compareidx]))
#maxError = np.max(diff)
#print(f'Max error between computed phi_0 and actual phi_0 at t=10 = {maxError}')

#%% Find time evolution of OBDM

# Get slater determinant
slater = getSlater(phi0_time, phi1_time) # slater[time idx][x1 idx][x2 idx]
mapping = np.zeros((numpts,numpts))
for i in range(numpts):
    for j in range(numpts): 
        mapping[i][j] = x[j] - x[i]
mapping = np.sign(mapping)
boseSlater = slater*mapping

# Integrate over second parameter for obdm
begin = time.time()
obdm = getOBDM(boseSlater)
end = time.time()

#%%
time_idx = 0
rsdp = np.array([obdm[time_idx, j, j] for j in range(numpts)])
actual = np.exp(-x**2)*(1+2*x**2)/(2*np.sqrt(np.pi))
print(f'Average difference between fermi OBDM and computed OBDM: {np.sum(np.abs(rsdp-actual))/numpts}')
print(f'Time to compute OBDM: {end - begin}')

fig6, ax6 = plt.subplots()
ax6.plot(x[start_idx:end_idx], 2*rsdp[start_idx:end_idx], label='Computed')
ax6.plot(x[start_idx:end_idx], 2*actual[start_idx:end_idx], label='Unperturbed')
ax6.set_title('Real Space Density Profile' + f' (t = {time_vector[time_idx]})' + r' ($\alpha = $' + f'{alpha},'  + r' $\lambda = $' + f'{lam})')
ax6.legend(loc='upper right')
ax6.grid()

normCons = np.einsum('ijj',np.abs(obdm))*h

fig8, ax8 = plt.subplots()
ax8.plot(time_vector, normCons )
ax8.set_title('Particle Number Conservation of OBDM vs. Time')
ax8.set_ylabel('Particle Number')
ax8.set_xlabel('Time')
ax8.grid()

#%% Find Momentum Space Density Profile
begin = time.time()
msdp = getMSDP(obdm)
end = time.time()
#%%
#start_idx = np.argmin(np.abs(x+plot_length/2.))
#end_idx = np.argmin(np.abs(x-plot_length/2.))
print(f'Time to compute MSDP: {end - begin}')
rsdp = np.array([obdm[0, j, j] for j in range(numpts)]) # this gives the inital RSDP for the bosons
fermiRSDP = np.array([obdm[0, j, j] for j in range(numpts)])

# We also need to find the initial MSDP of the fermions to compare to the final MSDP of bosons
fermiOBDM = getOBDM(slater)
fermiRSDP = np.array([fermiOBDM[0, j, j] for j in range(numpts)])
fermiMSDP = getMSDP(fermiOBDM)

#%%
time_idx = 1
start_idx = np.argmin(np.abs(x+10/2.))
end_idx = np.argmin(np.abs(x-10./2.))
fig7, ax7 = plt.subplots()
#ax7.plot(p[start_idx:end_idx], np.abs(msdp[0,start_idx:end_idx]), label=f'n(p; t={time_vector[0]}), ' + r' $\lambda = $' + f'{lam})')
ax7.plot(p[start_idx:end_idx], msdp[1,start_idx:end_idx], label=f'n(p; t={time_vector[1]})')
ax7.plot(p[start_idx:end_idx], msdp[2,start_idx:end_idx], label=f'n(p; t={time_vector[2]})')
ax7.plot(p[start_idx:end_idx], 2*rsdp[start_idx:end_idx], label='Initial Boson RSDP')
ax7.plot(p[start_idx:end_idx], 2*actual[start_idx:end_idx], label='Harmonic RSDP')
ax7.plot(p[start_idx:end_idx], np.abs(fermiMSDP[0,start_idx:end_idx]), label='Fermi MSDP')
ax7.set_title('Momentum Space Density Profile'  + r' ($\alpha = $' + f'{alpha},'  + r' $\lambda = $' + f'{lam})')
ax7.set_xlabel('p')
ax7.set_ylabel('n(p)')
ax7.legend(loc='upper right')
ax7.grid()


plt.show()