import torch 
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
import matplotlib.pyplot as plt
import numpy as np 
import time
# Test spline 
horizon = 100
num_samples = 2000


x = torch.linspace(0, 10, 10).to("cuda")
shift_time = 1.0

y = torch.repeat_interleave(torch.sin(x).reshape((1, 10, 1)), num_samples, 0).to("cuda")

now = time.time()
coeffs = natural_cubic_spline_coeffs(x, y)

spline = NaturalCubicSpline(coeffs)

res = spline.evaluate(torch.linspace(0, 10, horizon).to("cuda") + 1.0).cpu().numpy()
print("time is ", time.time() - now)
plt.plot(np.linspace(0, 10, horizon), res[0,:,0])
plt.plot(x.cpu().numpy(), y[0,:,0].cpu().numpy(), 'r*')
plt.savefig("dummy_name2.png")
# plt.show()
