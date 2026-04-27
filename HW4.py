import numpy as np
Ri = 0.03
Ro = 0.045
ho = 15
Tw = 40
Ta = 5
k = 50

X = np.array([[-Ro,0],
              [-Ro/np.sqrt(2), Ro/np.sqrt(2)],
              [-Ri/np.sqrt(2), Ri/np.sqrt(2)],
              [-Ri,0]])
conn = [[3,2,0],
         [1,0,2]]

dofs = np.array([1, 0, 3, 2])
print(f"X = \n{X}\n")

# element 1

J = (np.array([[-1, -1],[1,0],[0,1]]).T@X[conn[0], :]).T
print(f"element 1: J =\n{J}\n")

gradN  = np.array([[-1, -1],[1,0],[0,1]])@np.linalg.inv(J)
print(f"element 1: gradN = \n{gradN}\n")
gradN1 = gradN

Ke = 0.5*np.linalg.det(J)*k*gradN@gradN.T
print(f"element 1: Ke = \n{Ke}\n")

K = np.zeros((4,4))
K_ = np.zeros((4,4))
K_[np.ix_(dofs[conn[0]], dofs[conn[0]])] = Ke
print(f"element 1: K = \n{K_}\n")
K = K + K_

print("\n==================\n==================")
# element 2

J = (np.array([[-1, -1],[1,0],[0,1]]).T@X[conn[1], :]).T
print(f"element 2: J = \n{J}\n")

gradN  = np.array([[-1, -1],[1,0],[0,1]])@np.linalg.inv(J)
print(f"element 2: gradN = \n{gradN}\n")
gradN2 = gradN

Ke = 0.5*np.linalg.det(J)*k*gradN@gradN.T
print(f"element 2: Ke = \n{Ke}\n")

K_ = np.zeros((4,4))
K_[np.ix_(dofs[conn[1]], dofs[conn[1]])] = Ke
print(f"element 2: K = \n{K_}\n")
K = K + K_

print(f"K = \n{K}")
print("\n==================\n==================")
# film_BC

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