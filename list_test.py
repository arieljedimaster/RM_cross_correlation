import numpy as np

y = [[] for i in range(6)]

for i in np.arange(0,len(y)):
	y[0] += list(np.arange(0,i))

print y