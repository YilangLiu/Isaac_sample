import torch 
import time 
import matplotlib.pyplot as plt
import numpy as np 

def get_foot_step(duty_ratio, cadence, amplitude, phases, time):
    """
    Compute the foot step height.
    Args:
        amplitude: The height of the step.
        cadence: The cadence of the step (per second).
        duty_ratio: The duty ratio of the step (% on the ground).
        phases: The phase of the step. Warps around 1. (N-dim where N is the number of legs)
        time: The time of the step.
    """

    def step_height(t, footphase, duty_ratio):
        angle = (t + torch.pi - footphase) % (2 * torch.pi) - torch.pi
        angle = torch.where(duty_ratio < 1, angle * 0.5 / (1 - duty_ratio), angle)
        clipped_angle = torch.clip(angle, -torch.pi / 2, torch.pi / 2)
        value = torch.where(duty_ratio < 1, torch.cos(clipped_angle), 0)
        final_value = torch.where(torch.abs(value) >= 1e-6, torch.abs(value), 0.0)
        return final_value
    # time = torch.arange(100) / 100
    # time = time.to("cuda")
    h_steps = amplitude * torch.vmap(step_height, in_dims=(None, 0, None))(
        time * 2 * torch.pi * cadence + torch.pi,
        2 * torch.pi * phases,
        duty_ratio,
    )
    return h_steps


_gait_params = {
            #                  ratio, cadence, amplitude
            "stand": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32, device="cuda"),
            "walk": torch.tensor([0.75, 1.0, 0.08], dtype=torch.float32, device="cuda"),
            "trot": torch.tensor([0.45, 2, 0.08], dtype=torch.float32, device="cuda"),
            "canter": torch.tensor([0.4, 4, 0.06], dtype=torch.float32, device="cuda"),
            "gallop": torch.tensor([0.3, 3.5, 0.10], dtype=torch.float32, device="cuda"),
        }

_gait_phase = {
            "stand": torch.zeros(4, dtype=torch.float32, device="cuda"),
            "walk": torch.tensor([0.0, 0.5, 0.75, 0.25], dtype=torch.float32, device="cuda"),
            "trot": torch.tensor([0.0, 0.5, 0.5, 0.0], dtype=torch.float32, device="cuda"),
            "canter": torch.tensor([0.0, 0.33, 0.33, 0.66], dtype=torch.float32, device="cuda"),
            "gallop": torch.tensor([0.0, 0.05, 0.4, 0.35], dtype=torch.float32, device="cuda"),
        }



duty_ratio, cadence, amplitude = _gait_params["trot"]
phase = _gait_phase["trot"]


t = torch.zeros((40000)).to("cuda") 

x = np.array(range(0, 100, 1)) / 100
y = []
dt = 1/ 100
get_foot_step_vmap = torch.vmap(get_foot_step, in_dims=(None, None, None, None, 0))
z_feet_tar = get_foot_step_vmap(duty_ratio, cadence, amplitude, phase, t)
now = time.time()
for _ in range(0, 100, 1):
    z_feet_tar = get_foot_step_vmap(duty_ratio, cadence, amplitude, phase, t)
    t += dt
    y.append(z_feet_tar[:, :, None])

print("time is ", time.time() - now)

leg_names = ["FL", "FR", "RL", "RR"]
y = torch.cat(y, dim=2)
y = y.cpu().numpy()[123].T
for i in range(4):
    plt.plot(x, y[:, i], label=f"{leg_names[i]}_z")
plt.legend()
plt.savefig("gait_scheduler2.png")
# t = torch.tensor([0], dtype=torch.float32).reshape((-1, 1))
# t = t.to("cuda")
# now = time.time()
# x = np.array(range(0, 100, 1)) / 100
# y = []
# dt = 1/ 100
# for _ in range(0, 100, 1):
#     z_feet_tar = get_foot_step(duty_ratio, cadence, amplitude, phase, t)
#     t += dt
#     y.append(z_feet_tar)

# print("time is ", time.time() - now)
# # import pdb; pdb.set_trace()
# res = torch.cat(y, dim=1).squeeze() # (4, 100)
# plt.plot(x, res.cpu().numpy().T)
# plt.show()