

import numpy as np

xe = np.array([[-0.577350269189626, -0.577350269189626],
            [-0.577350269189626, +0.577350269189626],
            [+0.577350269189626, -0.577350269189626],
            [+0.577350269189626, +0.577350269189626]])
W = np.array([1, 1, 1, 1])

Ri = 0.03
Ro = 0.045
ho = 15
Tw = 40
Ta = 5
k = 50
Dz =1

X = np.array([[-Ro,0],
              [-Ro/np.sqrt(2), Ro/np.sqrt(2)],
              [-Ri/np.sqrt(2), Ri/np.sqrt(2)],
              [-Ri,0]])
conn = [[2,1,0,3]]

dofs = np.array([1,0,3,2])
print(f"X = \n{X}\n")

# element 1
def N(xi, eta):
    return np.array([(1-xi)*(1-eta)/4, (1+xi)*(1-eta)/4, (1+xi)*(1+eta)/4, (1-xi)*(1+eta)/4])
def dN_dxi(xi, eta):
    return np.array([[-(1-eta)/4, -(1-xi)/4],
                     [(1-eta)/4, -(1+xi)/4],
                     [(1+eta)/4, (1+xi)/4],
                     [-(1+eta)/4, (1-xi)/4]])

Ke = np.zeros((4,4))
for qp in np.arange(xe.shape[0]):
    xi = xe[qp, 0]
    eta = xe[qp, 1]
    J = np.dot(X.T, dN_dxi(xi, eta))
    detJ = np.linalg.det(J)
    gradN = np.dot(dN_dxi(xi, eta), np.linalg.inv(J))
    Ke = Ke + k * Dz * np.dot(gradN, gradN.T) * detJ * W[qp]

K = np.zeros((4,4))
K[np.ix_(dofs[conn[0]], dofs[conn[0]])] = Ke
print(f"element 1: K = \n{K}\n")



he = np.linalg.norm(X[1,:]-X[0,:])
H = ho*he/6*np.array([[2,1],[1,2]])
L = ho*he*Ta/2*np.array([[1],[1]])

print(f"film_BC: he = {he}\n")
print(f"film_BC: H = \n{H}\n")
print(f"film_BC: L = \n{L}\n")

K[np.ix_(dofs[[0,1]], dofs[[0,1]])] += H
print(f"K = \n{K}")
F = np.zeros((4,1))
F[dofs[[0,1]]] = L
print(f"F = \n{F}")

K_ff = K[np.ix_(dofs[[0,1]], dofs[[0,1]])]
F_ff = F[dofs[[0,1]]]
Kdd  = K[np.ix_(dofs[[2,3]], dofs[[2,3]])]
F_dd = F[dofs[[2,3]]]
K_fd = K[np.ix_(dofs[[0,1]], dofs[[2,3]])]

U = np.linalg.solve(K_ff, F_ff - K_fd@np.array([[Tw],[Tw]]))
print(f"U = \n{U}")

T = np.array([[U[0,0]],[U[1,0]],[Tw],[Tw]])

print("\n==================\n==================")

gradT1 = gradN1.T@T[conn[0]]
gradT2 = gradN2.T@T[conn[1]]
print(f"gradT1 = \n{gradT1}\n")
print(f"gradT2 = \n{gradT2}\n")

flux1 = -k*gradT1
flux2 = -k*gradT2
print(f"flux1 = \n{flux1}\n")
print(f"flux2 = \n{flux2}\n")