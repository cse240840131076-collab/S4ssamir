import numpy as np

arr = np.array([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]])

print("Original Array:")
print(arr)

print("1. Shape:", arr.shape)
print("2. Size:", arr.size)
print("3. Dimensions:", arr.ndim)
print("4. Data Type:", arr.dtype)
print("5. Sum:", np.sum(arr))
print("6. Mean:", np.mean(arr))
print("7. Maximum:", np.max(arr))
print("8. Minimum:", np.min(arr))
print("9. Standard Deviation:", np.std(arr))
print("10. Transpose:\n", np.transpose(arr))
print("11. Flatten:\n", arr.flatten())
print("12. Reshape:\n", arr.reshape(1, 9))
print("13. Square Root:\n", np.sqrt(arr))
print("14. Sort:\n", np.sort(arr))
print("15. Unique Values:\n", np.unique(arr))