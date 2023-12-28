import numpy as np
input_dim = 5
hidden_dims = [10,15,20,25]
num_classes = 30
for l, (i,j) in enumerate(zip([input_dim, *hidden_dims], [*hidden_dims, num_classes])) :
    print(l, i, j)