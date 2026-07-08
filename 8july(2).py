import numpy as np

arr = np.array([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]])

# Access values using index numbers
print("Element at [0][1]:", arr[0][1])
print("Element at [2][2]:", arr[2][2])

# Mathematical operations
print("Addition:", arr[0][0] + arr[1][1])
print("Subtraction:", arr[2][2] - arr[0][2])
print("Multiplication:", arr[1][0] * arr[0][1])
print("Division:", arr[2][1] / arr[1][1])
