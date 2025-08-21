import numpy as np
from mpmath import fp
from scipy import sparse
from decimal import Decimal
import time
import itertools
from math import factorial
from datetime import datetime

from scipy.constants import c as c_m_s, hbar as hbar_Js, m_e as me_kg
from DataHelpers.helpers import *

# Calculate kraus operator k of cavity photons rho_0 for 1 electron interaction
# with coupling constant gQ
def A_k(mk2, gQ1, gQ2, rho_shape, k, phi=0):
    """
    mk2: The number of +- k2 to consider in the summation. Going beyond
         the maximum fock state will yield no improvement
    gQ: cavity coupling constant (of both cavities)
    rho_0: initial density matrix of the two-cavity photon states.
            Axis 0 is photon state 1, Axis 1 is photon state 2.
    k: The post-selected electron energy, E = E0 - hwk
    phi: The free space propagation constant, ...
    """
    split = np.zeros(rho_shape+(2*mk2-1,),dtype = 'object')

    # k2: the number of photons absorbed in cavity 2. It is formally bounded by +-infty, but
    # is currently constrained by the number of input states
    for k2ct, k2 in enumerate(range(-mk2+1 + int(k/2), mk2 + int(k/2))):
        k1 = k - k2
        
        if (-rho_shape[1] < k2 < rho_shape[1]) \
            and (-rho_shape[1] < k1 < rho_shape[1]):
            m11 = np.max([-k1,0])
            m22 = np.max([-k2,0])

            # Calculates the matrix elements of the displacement operators
            # only if the operator takes density matrix members to density matrix members
            for nr in range(rho_shape[1]):
                if rho_shape[1] > nr+k1 > -1:
                    m_track = fp.hyp2f2(a1=1, a2=1+nr+k1+m11, b1=1+m11, b2=1+k1+m11, z=-gQ1**2)
                    m_track = m_track * (-1)**m11 * gQ1**(k1+2*m11) * fp.fac(nr+k1+m11)
                    m_track = m_track / fp.sqrt(fp.fac(nr)*fp.fac(nr+k1)) / fp.fac(m11) / fp.fac(k1+m11)
                    split[0,nr+k1,nr,k2ct]+= m_track * fp.exp(-1j*phi*k1**2)
                    
                if rho_shape[1] > nr+k2 > -1:
                    m_track = fp.hyp2f2(a1=1, a2=1+nr+k2+m22, b1=1+m22, b2=1+k2+m22, z=-gQ2**2)
                    m_track = m_track * (-1)**m22 * gQ2**(k2+2*m22) * fp.fac(nr+k2+m22)
                    m_track = m_track / fp.sqrt(fp.fac(nr)*fp.fac(nr+k2)) / fp.fac(m22) / fp.fac(k2+m22)
                    split[1,nr+k2,nr,k2ct]+=m_track
                    
    return np.array([fp.exp(fp.power(gQ1, 2) / fp.mpf('2')) * split[0],
                     fp.exp(fp.power(gQ2, 2) / fp.mpf('2')) * split[1]])

# Converts the Krause operators into tensor product form
def krause_toTProd(rho_0, mk2, k_range, k_mats, rhoNotTProd=True, checks=False):
    '''
    rho_0: initial density matrix of the two-cavity photon states. [rho_1, rho_2]
            Axis 0 is photon state 1, Axis 1 is photon state 2.
    mk2: The number of +- k2 to consider in the summation. Going beyond
            the maximum fock state should yield no improvement
    k_range: The range of k values to consider
    k_mats: The Krause operators for each k value
    rhoNotTProd: Whether rho_0 is in tensor product form or not. True if not
    '''
    if rhoNotTProd:
        A_k = np.zeros((np.kron(rho_0[0],rho_0[1])).shape + (len(k_range),), dtype='object')

    else:
        A_k = np.zeros(rho_0.shape + (len(k_range),), dtype='object')

    start=time.time()
    
    for k_ct, k in enumerate(k_range):
        for j in range(2*mk2-1):
            a = sparse.csr_matrix(k_mats[0,:,:,j,k_ct].astype('complex'))
            b = sparse.csr_matrix(k_mats[1,:,:,j,k_ct].astype('complex'))
            A_k[:,:,k_ct] += np.array(sparse.kron(a, b).todense())

    end=time.time()

    if checks:
        print("Tensor product Krause operators evaluated in {} seconds".format(end - start))

    return A_k

# Applies the kraus operators to the initial state rho_0
# In tensor product form
def krause_apply(rho_0, k_range, A_k, N, rhoNotTProd=True, checks=False):
    '''
    rho_0: initial density matrix of the two-cavity photon states. 
            Option A: Array of shape (2, n_fock_states, n_fock_states)
                        [rho_1, rho_2]
                        Axis 0 is photon state 1, Axis 1 is photon state 2.
            Option B: Array of shape (n_fock_states^2, n_fock_states^2)
                        Tensor product rho_1 x rho_2
    k_range: The range of k values to consider
    A_k: The Krause operators for each k value. In tensor product form already
    N: The number of applications of the Krause operators. Corresponds to the number of electrons
    checks: Whether to print out trace checks. Should be supressed for many iterations
    rhoNotTProd: Whether rho_0 is in tensor product form or not. 'True' if rho_0 is not
                in tensor product form.

    returns: The evolved density matrix (in tensor product form) and the diagonal of the 
             density matrix, which in ret_shape, is the cavity1-cavity2 photonic probability
             distribution.
    '''
    if rhoNotTProd:
        prob_m = sparse.csr_matrix(np.kron(rho_0[0], rho_0[1]))
        ret_shape = (rho_0.shape[1], rho_0.shape[1])
    else: 
        prob_m = sparse.csr_matrix(rho_0)
        ret_shape = (int(np.sqrt(rho_0.shape[0])), int(np.sqrt(rho_0.shape[0])))

    prob_mp1 = sparse.csr_matrix(np.zeros_like(prob_m, dtype='complex'))

    start=time.time()
    
    for n in range(N):
        for k_ct, k in enumerate(k_range):
            a = sparse.csr_matrix(np.array(A_k[:,:,k_ct],dtype='complex'))
            if k_ct==0:
                prob_mp1 = (a @ prob_m @ (np.conj(a)).T)
            else:
                prob_mp1 = prob_mp1 + (a @ prob_m @ (np.conj(a)).T)
            
        prob_m = prob_mp1
        prob_mp1 = sparse.csr_matrix(np.zeros_like(prob_m))

    end=time.time()

    prob_m = prob_m.todense()

    # prob_stat = np.reshape(np.diagonal(prob_m@np.conj(prob_m.T)), ret_shape)
    # prob_stat = np.reshape(np.diagonal(np.sqrt(prob_m@np.conj(prob_m.T))), ret_shape)
    # prob_stat = np.reshape(np.diagonal(np.abs(prob_m)), ret_shape)
    prob_stat = np.reshape(np.diagonal(prob_m), ret_shape)
    
    if checks:
        print("Done in {:.2f} minutes".format((end-start) / 60))
        print("Trace checks: {:.3f}".format(float(fp.nstr(np.sum(prob_stat)))))
    
    return prob_m, prob_stat

def coherentStateGenerate(alpha,n,dec=False,mps=False):
    """
    alpha: coherent parameter
    n: number of states to include in return
    """
    if dec:
        f = Decimal(np.exp(-alpha**2 / 2))
        ret = np.zeros((n,),dtype='object')

        for i in range(n):
            ret[i] = Decimal(alpha**(i) / np.factorial(i)**.5)
    elif mps:
        f = fp.sqrt(fp.exp(-fp.power(alpha, '2')))
        ret = np.zeros((n,),dtype='object')

        for i in range(n):
            ret[i] = fp.power(alpha,i) / fp.sqrt(fp.fac(i))
        
    else:
        f = np.exp(-alpha**2 / 2)
        ret = np.zeros((n,),dtype='object')

        for i in range(n):
            ret[i] = alpha**(i) / factorial(i)**.5

    return f*ret

def singlemodesqueezedStateGenerate(r,phi,n,dec=False,mps=False):
    """
    r: magnitude of squeeze parameter
    phi: phase of squeeze C = r exp(i phi)
    n: number of states to include in return
    """
    # Return Decimal array
    if dec:
        f = Decimal((np.sqrt(np.cosh(r)))**(-1))
        ret = np.zeros((n,),dtype='object')

        for i in range(n):
            if i <= n//2:
                amp = np.sqrt(factorial(2*i)) / 2**i / factorial(i)
                ret[2*i] = Decimal((-np.exp(1j*phi) * np.tanh(r))**(i) * amp)
    # Return mpmath array
    elif mps:
        f = fp.power((fp.sqrt(fp.cosh(r))), '-1')

        ret = np.zeros((n,),dtype='object')

        for i in range(n):
            if i <= n//2:
                amp = fp.sqrt(factorial(2*i)) / fp.power(2, i) / fp.factorial(i)
                ret[2*i] = fp.power((-fp.exp(1j*phi) * fp.tanh(r)), i) * amp
    # Return numpy.complex array
    else:
        f = (np.sqrt(np.cosh(r)))**(-1)
        ret = np.zeros((n,),dtype='object')

        for i in range(n):
            if i <= n//2:
                amp = np.sqrt(factorial(2*i)) / 2**i / factorial(i)
                ret[2*i] = (-np.exp(1j*phi) * np.tanh(r))**(i) * amp

    return f*ret

def twomodesqueezedStateGenerate(r,phi,n,dec=False,mps=False):
    """
    r: magnitude of squeeze parameter
    phi: phase of squeeze C = r exp(i phi)
    n: number of states to include in return
    return: 2n x 2n (tensor product) matrix of squeezed states
    """
    # Return Decimal array
    if dec:
        f = Decimal((np.sqrt(np.cosh(r)))**(-1))
        ret = np.zeros((n,),dtype='object')

        for i in range(n):
            if i <= n//2:
                amp = np.sqrt(factorial(2*i)) / 2**i / factorial(i)
                ret[2*i] = Decimal((-np.exp(1j*phi) * np.tanh(r))**(i) * amp)
    # Return mpmath array
    elif mps:
        f = fp.power((fp.sqrt(fp.cosh(r))), '-1')

        ret = np.zeros((n,),dtype='object')

        for i in range(n):
            if i <= n//2:
                amp = fp.sqrt(factorial(2*i)) / fp.power(2, i) / fp.factorial(i)
                ret[2*i] = fp.power((-fp.exp(1j*phi) * fp.tanh(r)), i) * amp
    # Return numpy.complex array
    else:
        # Set up the density matrix
        psi = np.zeros((n,n),dtype='object')
        ret = np.kron(psi,psi)

        f = (np.cosh(r))**(-1)

        # Fill the initial state psi = |n1 n2> as a matrix
        for i in range(n):
            amp = 1
            psi[i,i] = (-np.exp(1j*phi) * np.tanh(r))**(i) * amp

        # Translate the initial state into a tensor product matrix
        # ret = a*b |n n><n' n'|
        for i in range(n):
            for j in range(n):
                ret[i*n + i, j*n + j] = psi[i,i]*np.conj(psi[j,j])

            # ret[i*n + i, i*n + i] = ((-np.exp(1j*phi) * np.tanh(r))**(i) * amp)

    return f**2 *ret

# Free space propagation operator. Return goes into an e^{-i * return * k^2}
def FS_phi(omega, v, z):
    '''
    omega: angular frequency of the photon
    v: relativistic velocity of the electron
    z: propagation distance of the electron between cavities
    '''
    gamma = 1/np.sqrt(1-(v/c_m_s)**2)
    zD = 4*np.pi*gamma**3 * me_kg * v**3 / hbar_Js / omega**2
    
    return 2*np.pi * z/ zD

# g2 spectrum for (variable) m electrons traversing 2 (fixed) cavities
def g2_spectrum(prob_m, min_samples, max_samples):
    res = []
    for n in range(min_samples, max_samples+1):
        # Get all unique combinations in the format ((n1,n2), prob)
        # multiply probability by n-factorial to get the actual probability of getting that g2
        full_prob_list = list(itertools.combinations(np.ndenumerate(prob_m), n))

        g2s = np.zeros((2, len(full_prob_list)))

        for ct, entry in enumerate(full_prob_list):
            entry = np.array(entry)
            prob_of_occurance = np.prod([row[1] for row in entry])
            avg_n1n2 = sum(np.prod(row[0]) for row in entry) / n
            avg_n1 = sum(row[0][0] for row in entry) / n
            avg_n2 = sum(row[0][1] for row in entry) / n

            if fcomp(avg_n1, 0, 1e-8) or fcomp(avg_n2, 0, 1e-8):
                if fcomp(avg_n1n2, 0, 1e-8):
                    g2s[0][ct] = prob_of_occurance
                    g2s[1][ct] = 1
                else:
                    print('?')
                    continue

            else:
                g2s[0][ct] = prob_of_occurance
                g2s[1][ct] = avg_n1n2/avg_n1/avg_n2
            
        res.append(g2s)

    return res

# Calculates the g2 from the final state density matrix
# Option to limit calculation to n photons
def g2_FromFinalState(rho, n):
    '''
    rho: final photon1-photon2 density matrix
    n: the maximum photon number which will be included in the calculation
    '''
    # If n is not given, assume n is the dimension of the density matrix
    if n is None:
        n = rho.shape[0]

    # Initialize variables for <n1*n2> and <n1><n2>
    n1n2 = 0
    n1n1 = 0
    n2n2 = 0

    for n1 in range(n+1):
        for n2 in range(n+1):
            n1n2 += n1 * n2 * rho[n1, n2]
            n1n1 += n1 * rho[n1, n2]
            n2n2 += n2 * rho[n1, n2]

    # Calculate the g2
    correlation = n1n2 / (n1n1 * n2n2)

    return correlation

# Calculate the analogue of the kraus operators for the second order correlation
def B_k(gQ, rho_0, k):
    """
    gQ: cavity coupling constant (of both cavities)
    rho_0: initial density matrix of the two-cavity photon states.
            Axis 0 is photon state 1, Axis 1 is photon state 2.
    k: The post-selected cavity energy, E = E0 - hwk
    """
    split = np.zeros(rho_0[0].shape, dtype = 'object')

    m11 = np.max([-k,0])

    # Calculates the matrix elements of the displacement operators
    # only if the operator takes density matrix members to density matrix members
    for nr in range(rho_0[0].shape[1]):
        if rho_0.shape[1] > nr+k > -1:
            m_track = fp.nsum(lambda r: (-1)**r * gQ**(2*r+k) * fp.fac(nr+k+r) \
                                        / fp.fac(r) / fp.fac(r+k) \
                                        / fp.sqrt(fp.fac(nr)) / fp.sqrt(fp.fac(nr+k))  ,
                              [m11,fp.inf], method='shanks')
            split[nr+k,nr]+= m_track
                    
    return fp.exp(fp.power(gQ, 2) / fp.mpf('2')) * split

def B_kprime(gQ, rho_0, k):
    """
    gQ: cavity coupling constant (of both cavities)
    rho_0: initial density matrix of the two-cavity photon states.
            Axis 0 is photon state 1, Axis 1 is photon state 2.
    k: The post-selected cavity energy, E = E0 - hwk
    """
    split = np.zeros(rho_0[0].shape, dtype = 'object')

    m11 = np.max([-k,0])

    # Calculates the matrix elements of the displacement operators
    # only if the operator takes density matrix members to density matrix members
    for nr in range(rho_0[0].shape[1]):
        if rho_0.shape[1]-1 > nr+k-1 > -1:
            m_track = fp.nsum(lambda r: (-1)**r * gQ**(2*r+k) * fp.fac(nr+k+r) \
                                        / fp.fac(r) / fp.fac(r+k) \
                                        / fp.sqrt(fp.fac(nr)) / fp.sqrt(fp.fac(nr+k-1))  ,
                              [m11,fp.inf], method='shanks')
            split[nr+k,nr]+= m_track
                    
    return fp.exp(fp.power(gQ, 2) / fp.mpf('2')) * split

def Bsum(B_k, k_range, gQ1, rho0):
    Bsum = np.zeros_like(k_range, dtype='object')
    
    for k_ct, k in enumerate(k_range):
        Bsum[k_ct] = np.trace(rho0[0] @ np.conj(B_k(gQ1, rho0, k+1).T) @ B_k(gQ1, rho0, k))
        
    return Bsum

def mean_adag_0(rho0, p_i):
    adag0 = fp.mpc('0')
    for n in range(rho0[p_i].shape[0]-1):
        adag0 += fp.sqrt(n+1) * rho0[p_i][n+1,n]
        
    return adag0

def meanN_0(rho0, p_i):
    n0 = fp.mpc('0')
    for n in range(rho0[p_i].shape[0]):
        n0 += n * rho0[p_i][n,n]
        
    return n0

def meanN1N2_0(rho0):
    return meanN_0(rho0, 0)*meanN_0(rho0, 1)

def meanN1_1(rho0, gQ):
    return meanN_0(rho0,0) + np.abs(gQ)**2
    
def meanN2_1(rho0, gQ1, gQ2, phi, k_range, Bsum):
    ez = meanN_0(rho0,1) + np.abs(gQ2)**2
    hd = fp.mpc('0')
    hd_1 = mean_adag_0(rho0,1)
    
    for k_ct, k in enumerate(k_range):
        hd += fp.exp(1j*phi*(2*k+1)) * Bsum[k_ct]
     
    hd = 2 * gQ2 * fp.re( hd_1 * hd )
    
    return ez + hd
    

def meanN1N2_1(rho0, gQ1, gQ2, phi, k_range, Bsum_reg, Bsum_pri):
    ez = meanN1N2_0(rho0) + gQ2**2 * meanN_0(rho0,0) +\
                            gQ1**2 * meanN_0(rho0,1) +\
                            np.abs(gQ1*gQ2)**2
    
    hd = fp.mpc('0')
    hd_1 = mean_adag_0(rho0,1)
    
    for k_ct, k in enumerate(k_range):
        hd += fp.exp(1j*phi*(2*k+1)) * Bsum_pri[k_ct]
     
    hd = 2 * gQ2 * fp.re( hd_1 * hd )
    
    return (ez + hd) / (meanN1_1(rho0, gQ1) * meanN2_1(rho0, gQ1, gQ2, phi, k_range, Bsum_reg))


def g2_sampler(rho, m_measurements, n_times):
    '''
    g2_sampler: Samples the probability density m times and returns the g2 value for the m
                measurements. This process is repeated n_times and the spectrum of resulting
                g2 values is collected & returned.
    rho: The probability density matrix of the 2 photon system.
    m_measurements: The number of measurements to take for each run.
    n_times: The number of times to repeat the sampling process.
    '''
    # Sample the probability density m times
    samples = np.random.choice(np.arange(len(rho.flatten())), 
                               size=(n_times, m_measurements), 
                               p=rho.flatten())

    # Convert the samples to the indices of the density matrix
    # Calculate the average n1, n2, and n1*n2
    avg_n1 = np.mean(samples // rho.shape[1], axis=1)
    avg_n2 = np.mean(samples % rho.shape[1], axis=1)
    avg_n1n2 = np.mean( (samples // rho.shape[1]) * (samples % rho.shape[1]) )

    g2s = np.where((np.isclose(avg_n1, 0, atol=1e-16) \
                    | np.isclose(avg_n2, 0, atol=1e-16)) \
                    & np.isclose(avg_n1n2, 0, atol=1e-16), 
                    1,
                    avg_n1n2 / (avg_n1 * avg_n2))

    return g2s

def qmi_sampler(rho, m_measurements, n_times):
    '''
    qmi_sampler: Samples the probability density m times and returns the quantum mutual information
                 value for the m measurements. This process is repeated n_times and the spectrum of 
                 resulting QMI values is collected & returned.
    rho: The probability density matrix of the 2 photon system.
    m_measurements: The number of measurements to take for each run.
    n_times: The number of times to repeat the sampling process.
    '''

    qmis = np.zeros(n_times)

    for i in range(n_times):
        # Sample the probability density m times
        samples = np.random.choice(np.arange(len(rho.flatten())),
                                size=(m_measurements),
                                p=rho.flatten())

        # Construct a density matrix based on the samples
        rho_new, _ = np.histogram(samples, bins=np.arange(rho.size+1), density=True)

        # Reshape rho_new to the shape of rho
        rho_new = rho_new.reshape(rho.shape)

        # Compute the partial density matrices
        ### Do I need this??
        # pA = np.sum(rho_new, axis=1)
        # pB = np.sum(rho_new, axis=0)
    
        # Compute the quantum mutual information
        qmis[i] -= np.sum(np.multiply(rho_new, np.log2(rho_new), 
                                     where=(np.isclose(rho_new, 0, atol=1e-16)!=True)))

    return qmis

# Calculate the quantum mutual information
def QMI(rho, n):
    '''
    rho: final photon1-photon2 density matrix
    n: the maximum photon number which will be included in the calculation
    '''
    # If n is not given, assume n is the dimension of the density matrix
    if n is None:
        n = rho.shape[0]

    # Compute the eigenvectors and eigenvalues of rho_new
    eigvals, eigvecs = np.linalg.eig(rho[:n,:n])

    # Create the diagonal matrix
    rho_prime = np.diag(eigvals)

    # Compute the quantum mutual information
    qmi = - np.trace(np.multiply(rho_prime, np.emath.log(rho_prime), 
                                out=np.zeros_like(rho_prime),
                                where=(np.isclose(rho_prime, 0, atol=1e-16)!=True)))


    return qmi

def n2_brute(rhof, n):
    """
    n2_brute: Brute force calculation of the average number of photons in 
              cavity 2.
    rhof: The density matrix of the two cavity photon states
    n: The maximum photon number to consider
    """
    mean_n2 = 0
    
    for n1 in range(n+1):
        for n2 in range(n+1):
            mean_n2 += n2*rhof[n1,n2]
            
    return mean_n2

def threeMeasSamp(rho, m_measurements, n_times):
    '''
    threeMeasSamp: Samples the probability density m times
    rho: [(nxn), num_Es] The probability density matrix of the 2 photon system.
    m_measurements: The number of measurements to take for each run.
    n_times: The number of times to repeat the sampling process.
    '''
    # meas: [g2, n2, QMI]
    # print('here', flush=True)
    meas = np.zeros((3,n_times))

    # Sample the probability density m times
    samples = np.random.choice(np.arange(len(rho.flatten())), 
                            size=(n_times, m_measurements), 
                            p=np.abs(rho).flatten())

    # Convert the samples to the indices of the density matrix
    # Calculate the average n1, n2, and n1*n2
    avg_n1 = np.mean(samples // rho.shape[1], axis=1)
    avg_n2 = np.mean(samples % rho.shape[1], axis=1)
    avg_n1n2 = np.mean( (samples // rho.shape[1]) * (samples % rho.shape[1]) )

    g2s = np.where((np.isclose(avg_n1, 0, atol=1e-16) \
                    | np.isclose(avg_n2, 0, atol=1e-16)) \
                    & np.isclose(avg_n1n2, 0, atol=1e-16), 
                    1, avg_n1n2 / (avg_n1 * avg_n2))

    # Record g2
    meas[0,:] = g2s

    # Record avg_n2
    meas[1,:] = avg_n2

    # Calculate the quantum mutual information
    for i in range(n_times):
        # Construct a density matrix based on the samples
        rho_new, _ = np.histogram(samples[i], bins=np.arange(rho.size+1), density=True)

        # Reshape rho_new to the shape of rho,
        # constructing a density matrix based on samples
        rho_new = rho_new.reshape(rho.shape) 

        # print("Prob check: ", np.sum(rho_new), flush=True)

        # Compute the eigenvectors and eigenvalues of rho_new
        eigvals, eigvecs = np.linalg.eig(rho_new)

        # Create the diagonal matrix
        rho_prime = np.diag(eigvals)

        # Compute the quantum mutual information
        # plogp = np.where(np.isclose(eigvals, 0., atol=1e-16), 0., eigvals*np.log2(eigvals))
        eigvals_trunc = eigvals[np.abs(eigvals) > 1e-16] # Deletes all zero entries in eigvals
        # print(eigvals_trunc, flush=True)
        eigvals_trunc = np.real(eigvals_trunc)
        plogp = eigvals_trunc*np.emath.log(eigvals_trunc)
        # print(plogp, flush=True)
        qmi = - np.sum(plogp)


        # qmi = - np.sum(np.multiply(eigvals, np.log2(eigvals, 
        #                                               out=np.zeros_like(eigvals), 
        #                                               where=(np.isclose(eigvals, 
        #                                                                 0., 
        #                                                                 atol=1e-16)!=True)), 
        #                             out=np.zeros_like(eigvals),
        #                             where=(np.isclose(eigvals, 0., atol=1e-16)!=True)))
        
        # qmi = - np.trace(np.multiply(rho_prime, np.log2(rho_prime), 
        #                             out=np.zeros_like(rho_prime),
        #                             where=(np.isclose(rho_prime, 0, atol=1e-16)!=True)))


        # log_rho = np.log2(rho_new, out=np.full_like(rho_new, c_m_s), 
        #                   where=(np.isclose(rho_new, 0, atol=1e-8)!=True))
        # qmi = - np.sum(rho_new @ log_rho, out=np.zeros_like(rho_new), 
        #                   where=(np.isclose(log_rho, c_m_s, atol=1e-8) != True))
        # qmi = - np.sum(rho_new @ log_rho, out=np.zeros_like(rho_new), 
        #                   where=(np.isclose(log_rho, c_m_s, atol=1e-8) != True))
        # qmi = -np.sum(rho_new)

        # Record QMI
        meas[2,i] = qmi

    return meas

# Function to calculate the probability density
def calc_prob_density(data, bins):
    # ndata = np.vstack([g2, n2, qmi]).T
    # print(ndata.shape)
    hist_3d, edges = np.histogramdd(
        data,
        bins=bins,
        density=True
    )
    return hist_3d, edges