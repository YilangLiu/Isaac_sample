def control_interpolations(torch_array: torch.Tensor, horizon: int=200, degree: int =3, ctrl_dt: float=0.1, shift_time: float=0.0):
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
            knot_array_x = np.linspace(0, horizon*ctrl_dt, num_knots)
            ctrl_array_x = np.linspace(0, horizon*ctrl_dt, horizon) + shift_time
            spline = make_interp_spline(knot_array_x, knot_array_y, k=degree)
            interp_y = spline(ctrl_array_x)
            interp_torch_array[i,:,j] = torch.as_tensor(interp_y, device=device, dtype=dtype)
    return interp_torch_array