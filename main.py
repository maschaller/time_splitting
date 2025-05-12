import numpy as np
import progressbar
import matplotlib.pyplot as plt
import scipy.integrate 
import scipy.sparse
import time
import sys
import threading
import control as ctrl

import warnings
warnings.filterwarnings("ignore")


mode = 'Wave'

print(mode)
if mode == 'oneD':
    n = 1
    m = 1
    p = 1
    A = scipy.sparse.eye(1)
    B = A
    C=A
    Id = A
    Idm = A
    x0 = 5*np.ones(n)
    
    T = 5
    N = 101
    
    num_splittings = 2 # (N-1 should be divisible by number of splittings)
    maxIter = 500
    mu = 1
    alpha = 1 # weighting for control  

if mode == 'random ODE':
    n = 100 # state space dimension
    m = 60  # input space dimension
    p = 30 # output space dimension
    A = scipy.sparse.random(n,n, density = 0.5,random_state=1) #(J-R)@Q
    B = scipy.sparse.random(n,m, density = 0.5,random_state=1)#np.array([[1],[0],[0]]) #np.eye(n) #
    #C = scipy.sparse.random(p,n, density = 1)#np.array([[0],[1],[0]]).T #np.eye(n)#B.T
    C = scipy.sparse.eye(n)
    Id = scipy.sparse.eye(n)
    Idm = scipy.sparse.eye(m)
    x0 = np.ones(n)
    
    T = 5 # time horizon
    N = 101 # number of total time discretization points

    num_splittings = 10 # (N-1 should be divisible by number of splittings)
    maxIter = 1000
    mu = 0.1
    alpha = 10 # weighting for control
    

if mode == 'PH ODE':
    n = 3;
    m = 1;
    J = np.array([[0, 0, 1],[0, 0, -1],[-1, 1, 0]]);
    R = np.array([[1, 1, 0],[1, 1, 0 ],[0, 0, 0]]);
    Q = np.eye(n)
    A = (J-R)@Q
    
    B = np.array([[1],[0],[0]]) #np.eye(n) #
    C = np.array([[0],[1],[1]]).T #np.eye(n)#B.T
    Id = scipy.sparse.eye(n)
    Idm = scipy.sparse.eye(m)
    x0 = np.array([1,2,1])
    
    T = 4 # time horizon
    N = 51 # number of total time discretization points

    num_splittings = 5 # (N-1 should be divisible by number of splittings)
    maxIter = 500
    mu = 10
    alpha = 10 # weighting for control


if mode == 'Wave':
    A = scipy.sparse.load_npz('mats/A_w.npz')
    B = scipy.sparse.load_npz('mats/B_w.npz')
    C = scipy.sparse.load_npz('mats/C_w.npz')
    Id = scipy.sparse.load_npz('mats/Mass_w.npz')
    n = A.shape[0]
    m = B.shape[1]
    x0 = np.ones(n)
    Idm = Id
    T = 5 # time horizon
    N = 21 # number of total time discretization points

    num_splittings = 5 # (N-1 should be divisible by number of splittings)
    maxIter = 300
    mu = 20
    alpha = 1e-1 # weighting for control
    print(n)
    print('N,M,mu', num_splittings, N,mu)

if mode == 'Wave_partialobs':
    A = scipy.sparse.load_npz('mats/A_w_po.npz')
    B = scipy.sparse.load_npz('mats/B_w_po.npz')
    C = scipy.sparse.load_npz('mats/C_w_po.npz')
    Id = scipy.sparse.load_npz('mats/Mass_w_po.npz')
    n = A.shape[0]
    m = B.shape[1]
    x0 = np.ones(n)
    Idm = Id
    T = 5 # time horizon
    N = 21 # number of total time discretization points

    num_splittings = 5 # (N-1 should be divisible by number of splittings)
    maxIter = 500
    mu = 10
    alpha = 1e-1 # weighting for control
    print(n)
    print('N,M,mu', num_splittings, N,mu)
    
if mode == 'Adv_Diff':
    A = scipy.sparse.load_npz('mats/A_ad.npz')
    B = scipy.sparse.load_npz('mats/B_ad.npz')
    C = scipy.sparse.load_npz('mats/C_ad.npz')
    Id = scipy.sparse.load_npz('mats/Mass_ad.npz')
    n = A.shape[0]
    m = B.shape[1]
    x0 = np.ones(n)
    Idm = Id
    T = 5 # time horizon
    N = 7 # number of total time discretization points

    num_splittings = 2 # (N-1 should be divisible by number of splittings)
    maxIter = 1000
    mu = 1
    alpha = 1 # weighting for control

if mode == 'Adv_Diff3':
    A = scipy.sparse.load_npz('mats/A_ad3.npz')
    B = scipy.sparse.load_npz('mats/B_ad3.npz')
    C = scipy.sparse.load_npz('mats/C_ad3.npz')
    Id = scipy.sparse.load_npz('mats/Mass_ad3.npz')
    n = A.shape[0]
    m = B.shape[1]
    x0 = np.ones(n)
    Idm = Id
    T = 5 # time horizon
    N = 13 # number of total time discretization points
    
    num_splittings = 6 # (N-1 should be divisible by number of splittings)
    maxIter = 300
    mu = 20
    alpha = 1e-1 # weighting for control
    print(n)
    print('N,M,mu', num_splittings, N,mu)
    
if mode == 'Adv_Diff3_coarse':
    A = scipy.sparse.load_npz('mats/A_ad3_coarse.npz')
    B = scipy.sparse.load_npz('mats/B_ad3_coarse.npz')
    C = scipy.sparse.load_npz('mats/C_ad3_coarse.npz')
    Id = scipy.sparse.load_npz('mats/Mass_ad3_coarse.npz')
    n = A.shape[0]
    m = B.shape[1]
    x0 = np.ones(n)
    Idm = Id
    T = 5 # time horizon
    N = 13 # number of total time discretization points
    
    
    num_splittings = 6 # (N-1 should be divisible by number of splittings)
    maxIter = 300
    mu = 20
    alpha = 1e-1 # weighting for control
    print(n)
    print('N,M,mu', num_splittings, N,mu)
    
if mode == 'Adv_Diff3_partialobs':
    A = scipy.sparse.load_npz('mats/A_ad3_po.npz')
    B = scipy.sparse.load_npz('mats/B_ad3_po.npz')
    C = scipy.sparse.load_npz('mats/C_ad3_po.npz')
    Id = scipy.sparse.load_npz('mats/Mass_ad3_po.npz')
    n = A.shape[0]
    m = B.shape[1]
    x0 = np.ones(n)
    Idm = Id
    T = 10 # time horizon
    N = 13 # number of total time discretization points

    
    num_splittings = 6 # (N-1 should be divisible by number of splittings)
    maxIter = 200
    mu = 1
    alpha = 1e-1 # weighting for control
    print(n)
    print('N,M,mu', num_splittings, N,mu)
    
h = T/(N-1) # time step size
t = np.linspace(0,T,N) # time grid


# set up full optimality system

def set_up_opt_cond(N):
    calA =  scipy.sparse.lil_matrix((n * (N-1),n * (N-1)), dtype = np.float64)
    calA[0:n,0:n] = Id+h*A
    for j in range(1,N-1):
        calA[j*n:(j+1)*n, (j-1)*n:j*n ] = -Id
        calA[j*n:(j+1)*n, j*n:(j+1)*n ] = Id+h*A
    # Off-Diagonal blocks
    if mode == 'Wave' or mode == 'Wave_partialobs' or mode == 'Adv_Diff3' \
    or mode == 'Adv_Diff3_partialobs' or mode == 'Adv_Diff3_coarse':
        BBs = scipy.sparse.kron(np.eye((N-1),dtype=int),h*B)
        CsC = scipy.sparse.kron(np.eye((N-1),dtype=int),h*C)    
    else:
        BBs = scipy.sparse.kron(np.eye((N-1),dtype=int),h*B@B.T)
        CsC = scipy.sparse.kron(np.eye((N-1),dtype=int),h*C.T@C)
    M = scipy.sparse.bmat([[-CsC,-calA.T],[calA,-1/alpha*BBs]])
    return [calA,M]

# calA is the matrix corresponding to the ODE, M is the optimality system
[calA,M] = set_up_opt_cond(N)

# set up RHS
rhs = np.zeros( 2*n*(N-1), dtype = np.float64 )
rhs[n*(N-1): n*N] = Id@x0


# Compute optimal solution
Msparse = scipy.sparse.csc_matrix(M)

start = time.time()
zopt = scipy.sparse.linalg.spsolve(Msparse,rhs)
end = time.time()
print("Time for sparse solve of full system ", end - start)
print("Residual", np.linalg.norm(Msparse@zopt-rhs))


x = zopt[:n*(N-1)].reshape(N-1,n).T
lam = zopt[n*(N-1):].reshape(N-1,n).T

# compute optimal control
u = np.zeros((m,N-1))
for i in range(N-1):
    u[:,i] = 1/alpha*scipy.sparse.linalg.spsolve(Idm,B.T@lam[:,i])


######### splitted M ###########

def set_up_opt_M_j(N):
    calA =  scipy.sparse.lil_matrix((n * (N-1),n * (N-1)), dtype = np.float64)
    calA[0:n,0:n] = Id+h*A
    for j in range(1,N-1):
        calA[j*n:(j+1)*n, (j-1)*n:j*n ] = -Id
        calA[j*n:(j+1)*n, j*n:(j+1)*n ] = Id+h*A
    # Off-Diagonal blocks
    if mode == 'Wave' or mode == 'Wave_partialobs' or mode == 'Adv_Diff3' \
        or mode == 'Adv_Diff3_partialobs' or mode == 'Adv_Diff3_coarse':
        BBs = scipy.sparse.kron(np.eye((N-1),dtype=int),h*B.T)
        CsC = scipy.sparse.kron(np.eye((N-1),dtype=int),h*C) 
    else:
        BBs = scipy.sparse.kron(np.eye((N-1),dtype=int),h*B@B.T)
        CsC = scipy.sparse.kron(np.eye((N-1),dtype=int),h*C.T@C)
    dummy = scipy.sparse.lil_matrix((1,N-1), dtype = np.float64 )
    dummy[0,-1] = 1
    Er = scipy.sparse.kron(dummy,Id)
    El = scipy.sparse.kron(np.flip(dummy),Id)
    Mj = scipy.sparse.bmat([[-CsC,-calA.T,np.zeros((n*(N-1),n)),Er.T],\
                  [calA,-1/alpha*BBs,-El.T,np.zeros((n*(N-1),n))],\
                  [np.zeros((n,n*(N-1))),El,np.zeros((n,n)),np.zeros((n,n))],\
                   [-Er,np.zeros((n,n*(N-1))),np.zeros((n,n)),np.zeros((n,n))]])

    return scipy.sparse.lil_matrix(Mj)
    

Nj = int((N-1)/num_splittings)+1 # number of time steps per splitting interval
Mj = set_up_opt_M_j(Nj) # matrix for the subintervals
(Mjdim,a) = Mj.shape

# build M for forward application of muI+M
Mmat = scipy.sparse.lil_matrix(scipy.sparse.kron(scipy.sparse.eye(num_splittings),Mj))
Mmat[(Nj-1)*n:(Nj-1)*n+(Nj-1)*n, Mjdim-2*n:Mjdim-n] = 0 
Mmat[Mjdim-2*n:Mjdim-n, (Nj-1)*n:(Nj-1)*n+(Nj-1)*n] = 0 
Mmat[-n:,:] = 0
Mmat[:,-n:] = 0
Msparse = scipy.sparse.csc_matrix(Mmat)

# build N
Nmat = scipy.sparse.lil_matrix((num_splittings*Mjdim,num_splittings*Mjdim), dtype = np.float64)
for i in range(num_splittings-1):
    Nmat[(i+2)*Mjdim-2*n:(i+2)*Mjdim-n , (i+1)*Mjdim-n:(i+1)*Mjdim] =  -Id 
    Nmat[(i+1)*Mjdim-n:(i+1)*Mjdim, (i+2)*Mjdim-2*n:(i+2)*Mjdim-n] =  Id 
Nsparse = scipy.sparse.csc_matrix(Nmat) # convert to csc
 

# build M blockwise
Mup = Mj.copy()
Mup[(Nj-1)*n:(Nj-1)*n+(Nj-1)*n, Mjdim-2*n:Mjdim-n] = 0
Mup[Mjdim-2*n:Mjdim-n, (Nj-1)*n:(Nj-1)*n+(Nj-1)*n] = 0

Mlow = Mj.copy()
Mlow[-n:,:] = 0
Mlow[:,-n:] = 0 


Massoption = 2
##### First option: Just identity
if Massoption == 1:
    muId = mu * scipy.sparse.eye(Mjdim*num_splittings, dtype = np.float64, format = 'csc')

##### Riesz in L^2(0,T;L^2)
else:
    k1 = h * scipy.sparse.kron(scipy.sparse.eye(Nj-1),Id)
    muIdj = mu * scipy.sparse.csc_matrix(scipy.sparse.block_diag((k1,k1,Id,Id)))
    muId = scipy.sparse.csc_matrix(scipy.sparse.kron(scipy.sparse.eye(num_splittings),muIdj))
#####

start = time.time() # start timer for our method

# factorize muI+N
if mode == 'Adv_Diff3' or mode == 'Adv_Diff3_partialobs' or mode == 'Adv_Diff3_coarse':
    muId_N_fac = scipy.sparse.linalg.spilu(muId-Nsparse) 
else:
    muId_N_fac = scipy.sparse.linalg.splu(muId-Nsparse) 
        

#build diagonal blocks of muI+M (three different kinds)
Mjsparse = scipy.sparse.csc_matrix(Mj)
Mupsparse = scipy.sparse.csc_matrix(Mup)
Mlowsparse = scipy.sparse.csc_matrix(Mlow)

if Massoption == 1:
    muId_small = mu * scipy.sparse.eye(Mjdim, dtype = np.float64, format = 'csc')
else:
    muId_small = muIdj

#factorize diagonal blocks of muI+M
muId_Mj_fac = scipy.sparse.linalg.splu(muId_small-Mjsparse)
muId_Mup_fac = scipy.sparse.linalg.splu(muId_small-Mupsparse)
muId_Mlow_fac = scipy.sparse.linalg.splu(muId_small-Mlowsparse)
end = time.time()
factorization_time = end-start # calculate time for factorization
print("Time for factorizations ", factorization_time)

# solve muI+M sequential
def solveMpart(b):
    start = time.time()
    z = np.zeros(Mjdim*num_splittings, dtype = np.float64)
    z[0:Mjdim] = muId_Mup_fac.solve(b[0:Mjdim])
    for i in range(num_splittings-1):
        mysolve(b,z,i)
    z[-Mjdim:] = muId_Mlow_fac.solve(b[-Mjdim:])
    return z

# helper function for parallel solve
def mysolve(src,target,i):
    target[(i+1)*Mjdim:(i+2)*Mjdim] = muId_Mj_fac.solve(src[(i+1)*Mjdim:(i+2)*Mjdim])

# solve muI+M parallel
def solveMpart_parallel(b):
    start = time.time()
    z = np.zeros(Mjdim*num_splittings, dtype = np.float64)
    z[0:Mjdim] = muId_Mup_fac.solve(b[0:Mjdim])
    
    t = []
    for i in range(num_splittings-1):
        t.append(threading.Thread(target=mysolve, args=(b,z,i)))
        t[i].start()
    for i in range(num_splittings-1):
        t[i].join()

    z[-Mjdim:] = muId_Mlow_fac.solve(b[-Mjdim:])
    return z

# sequential method
def Peaceman_Rachfort(z):
    zp = solveMpart((muId+Nsparse)@muId_N_fac.solve((muId+Msparse)@z) - (muId+Nsparse)@muId_N_fac.solve(f) - f)
    return zp

# parallel method
def Peaceman_Rachfort_parallel(z):
    zp = solveMpart_parallel((muId+Nsparse)@muId_N_fac.solve((muId+Msparse)@z)-(muId+Nsparse)@muId_N_fac.solve(f) - f)   
    return zp

# compute rhs
f = np.zeros(Mjdim*num_splittings, dtype = np.float64)
f[(Nj-1)*n:(Nj-1)*n+n] = Id@x0
z = np.zeros((Mjdim*num_splittings,maxIter+1), dtype = np.float64)
start = time.time()

# main loop
for i in progressbar.progressbar(range(maxIter)):
    if i == 1:
        s = time.time()
        
    z[:,i+1] = Peaceman_Rachfort_parallel(z[:,i])
    
    if i == 1:
        print("time for one iteration", time.time()-s)

end = time.time()
print("Time for Peaceman-Rachfort iteration: ", end-start)
print("Considering also the factorizations: ", end-start + factorization_time)


# postprocessing and plotting
xs = np.zeros((maxIter,n,N-1))
ls = np.zeros((maxIter,n,N-1))
error = np.zeros((maxIter,3))

Nb = (Nj-1)*n
dummyvec1 = np.zeros((N-1)*n)
dummyvec2 = np.zeros((N-1)*n)

for j in range(0,maxIter):
    dummyvec1[0:Nb] = z[0:(Nj-1)*n,j]
    dummyvec2[0:Nb] = z[(Nj-1)*n:2*(Nj-1)*n,j]   
    for i in range(0,num_splittings):
        dummyvec1[i*Nb:(i+1)*Nb] = z[i*Mjdim:i*Mjdim+(Nj-1)*n,j]
        dummyvec2[i*Nb:(i+1)*Nb] = z[i*Mjdim+(Nj-1)*n:i*Mjdim+2*(Nj-1)*n,j]
    xs[j] = dummyvec1.reshape(N-1,n).T
    ls[j] = dummyvec2.reshape(N-1,n).T
    for k in range(N-1):
        error[j,0] = error[j,0] + h * np.sqrt((xs[j][:,k]-x[:,k]) @ Id @ (xs[j][:,k]-x[:,k]))
        error[j,1] = error[j,1] + h * np.sqrt((ls[j][:,k]-lam[:,k]) @ Id @ (ls[j][:,k]-lam[:,k]))
        ucurr = 1/alpha*scipy.sparse.linalg.spsolve(Idm,B.T@ls[j][:,k])
        error[j,2] = error[j,2] + h * np.sqrt((ucurr-u[:,k]) @ Idm @ (ucurr-u[:,k]))
    
        
# norm over time
normxtot = 0
normltot = 0
normutot = 0
normx = np.zeros((N-1))
normlam = np.zeros((N-1))
normu = np.zeros((N-1))
for i in range(N-1):
    normxtot = normxtot + h * np.sqrt(x[:,i]@Id@x[:,i])
    normltot = normltot + h * np.sqrt(lam[:,i]@Id@lam[:,i])
    normutot = normutot + h * np.sqrt(u[:,i]@Idm @ u[:,i])
    
    normx[i] = np.sqrt(x[:,i]@Id@x[:,i])
    normlam[i] = np.sqrt(lam[:,i]@Id@lam[:,i])
    normu[i] = np.sqrt(u[:,i]@Idm@u[:,i])
    
    
print("State error at end", error[-1,0]/normxtot)
print("Adjoint error at end", error[-1,1]/normltot)
print("Control error at end", error[-1,2]/normutot)

for i in range(maxIter):
    if error[i,2]/normutot < 1e-2:
        print("i", i)
        break

normxs = np.zeros((N-1))
normlams = np.zeros((N-1))
normus = np.zeros((N-1))
for i in range(N-1):
    normxs[i] = np.sqrt(xs[-1][:,i]@Id@xs[-1][:,i])
    normlams[i] = np.sqrt(ls[-1][:,i]@Id@ls[-1][:,i])
    ucurr = 1/alpha*scipy.sparse.linalg.spsolve(Idm,B.T@ls[-1][:,i])
    normus[i] = np.sqrt(ucurr@Idm@ucurr)

fig_soln, ax_soln = plt.subplots(3,1)#constrained_layout=True)
fig_soln.tight_layout(pad=1.3)
ax_soln[0].plot(t[1:], normxs, color='red', linestyle='solid', label='norm state from splitting')
ax_soln[0].plot(t[1:], normx, color='green', linestyle='solid', label='norm optimal state')
ax_soln[0].set_title("Norm of state")
ax_soln[0].legend()

ax_soln[1].plot(t[1:], normus, color='red', linestyle='solid', label='norm control from splitting')
ax_soln[1].plot(t[1:], normu, color='green', linestyle='solid', label='norm optimal control')
ax_soln[1].set_title("Norm of control")
ax_soln[1].legend()

ax_soln[2].plot(t[1:], normlams, color='red', linestyle='solid', label='norm control from splitting')
ax_soln[2].plot(t[1:], normlam, color='green', linestyle='solid', label='norm optimal control')
ax_soln[2].set_title("Norm of adjoint")
ax_soln[2].legend()
plt.xlabel('time')
plt.show()


# error over PR-Iterations
fig_soln, ax_soln = plt.subplots(3,1)#constrained_layout=True)
fig_soln.tight_layout(pad=1.3)

ax_soln[0].plot(error[:, 0]/normxtot, label='State')
ax_soln[0].set_title("rel error of state")
ax_soln[0].set_yscale('log')
ax_soln[0].set_yticks([1e-6,1e-4,1e-2,1e-1])

ax_soln[1].plot(error[:, 1]/normltot, label='Adjoint')
ax_soln[1].set_title("rel. error of adjoint")
ax_soln[1].set_yscale('log')
ax_soln[1].set_yticks([1e-6,1e-4,1e-2,1e-1])

ax_soln[2].plot(error[:, 2]/normutot, label='Control')
ax_soln[2].set_title("rel. error of control")
ax_soln[2].set_yscale('log')
ax_soln[2].set_yticks([1e-6,1e-4,1e-2,1e-1])
plt.xlabel('Iterations')
plt.show()


import pickle

with open('data/'+mode+'mu_'+str(mu)+'.pkl', 'wb') as f:
    pickle.dump([error,normxtot,normltot,normutot], f)





