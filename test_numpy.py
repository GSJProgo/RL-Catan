import numpy as np 

a = np.zeros((3,5))
b = np.ones((3,5))
d = np.zeros(8)
print(d)
c = a*b 
print(c)

a[0][0] = 1
c = a*b 
print(c)

a[1][0] = 2

c = a*b 
print(c)

d = 1 - c

print(d)

def g():
    print(np.random.choice(np.arange(1,6), p = (0.2,0.2,0.2,0.2,0.2)))

g()
g()
g()
g()
g()
g()
g()
g()
g()
g()
g()
g()
g()
g()
g()
g()
g()
g()