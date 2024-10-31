import numpy as np

input_dim = 3
h_dim = 5
out_dim = 3


x = np.random.rand(3)

print(x)

w1 = np.random.rand(input_dim, h_dim)
b1 = np.random.rand(h_dim)
w2 = np.random.rand(h_dim, out_dim)
b2 = np.random.rand(out_dim)



def relu(t):
    return np.maximum(0, t)

def softmax(x):
    print(f"вход в софтмакс:\n {x}")
    out = np.exp(x)
    print(f"экспонента:\n {out}")
    return out / np.sum(out)

t1 = x @ w1 +b1

print(f"x До: \n {x} \n веса: \n {w1} \n шаг: \n{b1}")
print(f"f1: \n {t1}")

h1 = relu(t1)

print(f"h1: \n {h1}")

t2 = h1 @ w2 + b2

print(f"t2: \n {t2}")

z = softmax(t2)
print(f"z: \n {z}")

