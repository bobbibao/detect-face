import numpy as np

arr = np.array([[[[[1, 2, 3],[1, 2, 3]],[[1, 2, 3],[1, 2, 3]]],[[[1, 2, 3],[1, 2, 3]],[[1, 2, 3],[1, 2, 3]]]],[[[[1, 2, 3],[1, 2, 3]],[[1, 2, 3],[1, 2, 3]]],[[[1, 2, 3],[1, 2, 3]],[[1, 2, 3],[1, 2, 3]]]]], ndmin=5)

print(arr)
print('number of dimensions :', arr.ndim)