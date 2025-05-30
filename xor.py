from nn import MLP
from helpers import draw_dot
from value import Value

n = MLP(3, [4, 4, 1])

xs = [
  [0.0, 0.0],
  [1.0, 0.0],
  [0.0, 1.0],
  [1.0, 1.0],
]
ys = [0.0, 1.0, 1.0, 1.0]

ypred = []

for k in range(200):
    # forward
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # backward
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.05 * p.grad
    
    print(k, loss.data)

print(f"answers: {ypred}")
