import numpy as np

def q_val(mu1, mu2, eps):
    return np.sqrt(np.dot(np.dot((mu1 - mu2).T, eps), (mu1 - mu2)))/2.0

print(q_val(np.array([0,0]),
            np.array([3,0]),
            np.array([[1,0],[0,1]])
     ))
print(q_val(np.array([-1,0]),
            np.array([1,0]),
            np.array([[4,3],[3,4]])
     ))
print(q_val(np.array([-1,0]),
            np.array([1,0]),
            np.array([[1,0],[0,1]])
     ))