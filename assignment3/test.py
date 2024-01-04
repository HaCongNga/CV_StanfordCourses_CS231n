import numpy as np
a = np.array([1,2,3])
b = np.array([[1], [2], [3]])
c = b ** b
print(a**a)
print(b**b)
print(b[2:, :])
print(c.sum())
print(b[0:-1, :])