
from functools import reduce
import numpy as np
a = [1,3,4,2]
a = np.array(a)
na = np.argsort(a)
c = [sum(a[na[:i]]) for i in range(len(a))]
# for i in range(len(c)):
# b = reduce(lambda x,y:x+y,c[i])
print(a[na[:3]] > 1)
print(np.argwhere(a[na[:3]] > 1))