from fenics import *
from fenics_adjoint import *
from collections import OrderedDict
import scipy.sparse.linalg
import scipy.sparse
import scipy.io

# Define a vector-valued expression
class MyVectorExpression(UserExpression):
    def eval(self, value, x):
        # Define the components of the vector
        value[0] = -0.5  # First component
        value[1] = 0 #x[0] - x[1]  # Second component
        value[2] = 0
    
    def value_shape(self):
        # Return the shape of the vector
        return (3,)

# Initialize the vector-valued expression
vector_expr = MyVectorExpression(degree=2)


b = MyVectorExpression(degree=2)
c = Expression("5", t = 0,degree = 0)
nu = Constant(1e-1)

n = 14
# Next, we define the discretization space:

mesh = BoxMesh(Point(0,0,0),Point(1,1,1), n-1, n-1, n-1)
#mesh = UnitCubeMesh(n,n,n)

eta = FacetNormal(mesh)
f = Expression('1', degree=0)

V = FunctionSpace(mesh, 'P', 1)
    
# Define boundary condition
u_D = Constant(0) #Expression('0', degree=0)
def boundary(x, on_boundary):
    #return on_boundary
    return
bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)


#     - c*u*v*dx\
#    - inner(b,grad(u))*v*dx\
F = ( nu*inner(grad(u), grad(v)))*dx\
    - f*v*dx
a, L = lhs(F), rhs(F)

# Assemble matrix
A = assemble(a)

load = assemble(L)
    
a1PET = as_backend_type(A).mat()
ANP1 = scipy.sparse.csr_matrix(a1PET.getValuesCSR()[::-1], shape=a1PET.size)
scipy.sparse.save_npz('../mats/A_ad3_po.npz', ANP1)


J = 0.5*(ANP1-ANP1.T)
R = 0.5*(ANP1+ANP1.T)

asd,fdg = scipy.sparse.linalg.eigs(R)
print(min(asd))
#print(scipy.sparse.linalg.cond(ANP1))

#ew1, ev = scipy.sparse.linalg.eigsh(ANP1, which='LM')
# ew2, ev = scipy.sparse.linalg.eigsh(ANP1, sigma=1e-8)   #<--- takes a long time

# ew1 = abs(ew1)
# ew2 = abs(ew2)

# condA = ew1.max()/ew2.min()
# print("condition number", condA)
# B, C and Mass Matrix for time stepping

tol = 0.5

observation_region = Expression('x[0] >=  tol ? 1 : 0', degree=0,tol=tol)
control_region = Expression('x[0] <=  tol ? 1 : 0', degree=0,tol=tol)     
 
mass = u*v*dx
b = u*v*control_region*dx
c = u*v*observation_region*dx 
 
B = assemble(b)
C = assemble(c)
Mass = assemble(mass)
 
BPET = as_backend_type(B).mat()
BNP = scipy.sparse.csr_matrix(BPET.getValuesCSR()[::-1], shape=BPET.size)

CPET = as_backend_type(C).mat()
CNP = scipy.sparse.csr_matrix(CPET.getValuesCSR()[::-1], shape=CPET.size)

MASSPET = as_backend_type(Mass).mat()
MassNP = scipy.sparse.csr_matrix(MASSPET.getValuesCSR()[::-1], shape=MASSPET.size)

scipy.sparse.save_npz('../mats/B_ad3_po.npz', BNP)
scipy.sparse.save_npz('../mats/C_ad3_po.npz', CNP)
scipy.sparse.save_npz('../mats/Mass_ad3_po.npz', MassNP)


#mdic = {"A": ANP1, "b": load.get_local().T, "C": CNP, "B": BNP, "Mass": MassNP}
#scipy.io.savemat('data/adv_diff3_51.mat', mdic)