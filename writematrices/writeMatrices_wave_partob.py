
from fenics import *
from dolfin import *

import scipy.sparse



MeshN = 11

mesh = UnitSquareMesh(MeshN, MeshN)
n = FacetNormal(mesh)

hx = 1/MeshN

V_h = FiniteElement('CG', mesh.ufl_cell(), 1)
Q_h = VectorElement('CG', mesh.ufl_cell(),  2)
W = FunctionSpace(mesh, V_h*Q_h)
V,Q = W.split()


V_collapse = V.collapse()


damping = Constant(0)
wavenumber = Constant(1)


(p, q) = TrialFunctions(W)
(v, w) = TestFunctions(W)

d = Function(V_collapse, name="data")


# Assemble A Part
a = (-damping*p*v - wavenumber*dot(q,grad(v))) * dx  \
    + wavenumber*dot(grad(p),w)* dx
#a = lhs(F), rhs(F)

A = assemble(a)

APET = as_backend_type(A).mat()
ANP1 = scipy.sparse.csr_matrix(APET.getValuesCSR()[::-1], shape=APET.size)
scipy.sparse.save_npz('../mats/A_w_po.npz', ANP1)

tolc = 0.5
tolo = 0.5

observation_region = Expression('x[0] >=  tol || x[1] >= tol ? 1 : 0', degree=0,tol=tolo)
control_region = Expression('x[0] <=  tol || x[1] <= tol ? 1 : 0', degree=0,tol=tolc)     
 
mass = (p*v + inner(q,w))* dx
b = (p*v)*control_region*dx
c = (inner(q,w)+p*v)*observation_region*dx 
 
B = assemble(b)
C = assemble(c)
Mass = assemble(mass)

BPET = as_backend_type(B).mat()
BNP = scipy.sparse.csr_matrix(BPET.getValuesCSR()[::-1], shape=BPET.size)
scipy.sparse.save_npz('../mats/B_w_po.npz', BNP)

CPET = as_backend_type(C).mat()
CNP = scipy.sparse.csr_matrix(CPET.getValuesCSR()[::-1], shape=CPET.size)
scipy.sparse.save_npz('../mats/C_w_po.npz', CNP)

MassPET = as_backend_type(Mass).mat()
MassNP = scipy.sparse.csr_matrix(MassPET.getValuesCSR()[::-1], shape=MassPET.size)
scipy.sparse.save_npz('../mats/Mass_w_po.npz', MassNP)

