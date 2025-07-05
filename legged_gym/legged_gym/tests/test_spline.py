import torch 
import matplotlib.pyplot as plt
import numpy as np  
import scipy.interpolate as si

def bezier_spline(torch_array: torch.Tensor, horizon=200, degree=2):
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
            knot_array_x = np.linspace(0, num_knots, num_knots)
            spline = si.splrep(knot_array_x, knot_array_y, k=degree)
            interp_array_x = np.linspace(0, num_knots, horizon)
            interp_array_y = si.splev(interp_array_x, spline)
            interp_torch_array[i,:,j] = torch.as_tensor(interp_array_y, device=device, dtype=dtype)
    return interp_torch_array


# Test spline 
horizon = 100
x = torch.linspace(0, 10, 10)
y = torch.sin(x)
res = bezier_spline(y.reshape((1, 10, 1)), horizon)
res = res.cpu().numpy()
plt.plot(np.linspace(0, 10, horizon), res[0,:,0])
plt.plot(x, y, 'r*')
plt.show()
