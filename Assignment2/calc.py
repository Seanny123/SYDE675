import numpy as np
from numpy.linalg import inv

def q_val_ged(mu1, x, eps):
    return np.sqrt(np.dot(np.dot((mu1 - x).T, inv(eps)), (mu1 - x)))

def q_val(mu1, x, eps):
    return np.sqrt((mu1 - x) * 1/eps * (mu1 - x))

print("MED and GED case 1")
print(q_val(3.0, 1.5, 1.0))
print("MED case 2")
print(q_val(1.0, 0.0, 4.0))
print("GED case 2")
print(q_val_ged(np.array([-1,0]),
            np.array([0,0]),
            np.array([[4,3],[3,4]])
     ))
print("MED case 3.1")
print(q_val(3.0, 1.5, 7.0))
print("MED case 3.2")
print(q_val(0.0, 1.5, 3.0))