import numpy as np

a = np.arange(0, 10, 1)
b = np.arange(10, 20, 1)
print(a, b)
# result:[0 1 2 3 4 5 6 7 8 9] [10 11 12 13 14 15 16 17 18 19]
state = np.random.get_state()
np.random.shuffle(a)
print(a)
# result:[6 4 5 3 7 2 0 1 8 9]
np.random.set_state(state)
np.random.shuffle(b)
print(b)
# result:[16 14 15 13 17 12 10 11 18 19]



