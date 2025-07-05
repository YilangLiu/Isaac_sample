import matplotlib.pyplot as plt
import numpy as np 
import torch
from scipy.interpolate import make_interp_spline
import time

def knots_to_control(torch_array: torch.Tensor, horizon=200, degree=3, dt=0.1):
    # Effictively upsampling the current control knots to actual
    # control sequence that is used for rollouts
    # torch_array (num_samples, num_knots, num_actions)
    device = torch_array.device
    dtype = torch_array.dtype
    num_samples= torch_array.shape[0]
    num_knots= torch_array.shape[1]
    num_actions = torch_array.shape[2]
    numpy_array = torch_array.cpu().numpy()
    interp_torch_array = torch.zeros((num_samples, horizon, num_actions), device=device, dtype=dtype)

    for i in range(num_samples):
        for j in range(num_actions):
            knot_array_y = numpy_array[i,:,j]
            knot_array_x = np.linspace(0, horizon*dt, num_knots)
            ctrl_array_x = np.linspace(0, horizon*dt, horizon)
            spline = make_interp_spline(knot_array_x, knot_array_y, k=degree)
            interp_y = spline(ctrl_array_x)
            interp_torch_array[i,:,j] = torch.as_tensor(interp_y, device=device, dtype=dtype)
    return interp_torch_array

# Test spline 
horizon = 100
knots = 10
x = torch.linspace(0, 10, horizon)
y = torch.sin(x)
now = time.time()
res = knots_to_control(y.reshape((1, horizon, 1)), knots)
print("time is ", time.time() - now)
res = res.cpu().numpy()
plt.plot(np.linspace(0, 10, knots), res[0,:,0], 'r*')
plt.plot(x, y)
plt.show()
