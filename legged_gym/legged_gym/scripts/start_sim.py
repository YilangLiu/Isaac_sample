import os 
os.environ['VK_ICD_FILENAMES'] ='/usr/share/vulkan/icd.d/nvidia_icd.json'
import isaacgym 
from legged_gym.envs import * 
from legged_gym.utils import get_args, task_registry
import torch 
import faulthandler
import pdb
from rsl_rl.planner.mppi import MPPI_Isaac
from multiprocessing import Process, shared_memory
import multiprocessing as mp 
import zerorpc
import matplotlib.pyplot as plt
import numpy as np 
import time 
import copy
from legged_gym.utils import webviewer

def Planner(args, env_cfg, planner_cfg, ready_event):
    ctrl_dt = env_cfg.control.decimation * env_cfg.sim.dt
    action_shm = shared_memory.SharedMemory(
        name="action_shm", create=False, size=env_cfg.env.num_agent_envs * planner_cfg.planner.horizon * env_cfg.env.num_actions * 32
    )
    action_shared = np.ndarray(
        (env_cfg.env.num_agent_envs, planner_cfg.planner.horizon, env_cfg.env.num_actions), dtype=np.float32, buffer=action_shm.buf
    )
    action_shared[:] = np.zeros((env_cfg.env.num_agent_envs, planner_cfg.planner.horizon, env_cfg.env.num_actions), dtype=np.float32)

    plan_time_shm = shared_memory.SharedMemory(
            name="plan_time_shm", create=False, size=32
        )
    plan_time_shared = np.ndarray(
        1, dtype=np.float32, buffer=plan_time_shm.buf
    )
    plan_time_shared[0] = -ctrl_dt

    time_shm = shared_memory.SharedMemory(
                name="time_shm", create=False, size=32
            )
    time_shared = np.ndarray(1, dtype=np.float32, buffer=time_shm.buf)
    time_shared[0] = 0.0
    
    last_plan_time = time_shared[0]
    extras = {}
    extras["ready_event"] = ready_event
    if planner_cfg.planner.name == "MPPI":
        planner = MPPI_Isaac(env_cfg, planner_cfg, args, extras)
    try:
        with torch.inference_mode():
            while True:
                plan_time = time_shared[0]
                shift_time = plan_time - last_plan_time
                # Add information are stored in the shared memory
                # now = time.time()
                actions = planner.update(shift_time)
                # planner.update(shift_time)
                # print("overall update freq is ", time.time() - now)
                # actions = torch.zeros((env_cfg.env.num_agent_envs, planner_cfg.planner.horizon, env_cfg.env.num_actions), dtype=torch.float32)
                # time.sleep(0.01)
                plan_time_shared[0] = plan_time
                last_plan_time = plan_time
                action_shared[:] = actions.cpu().numpy()
    except KeyboardInterrupt:
        print("Keyboard Terminated for Planning")
    else:
        # plt.ioff()
        # plt.show()
        print("Planner Process Terminated")

def World(args, env_cfg, planner_cfg, ready_event):
    if args.web:
        web_viewer = webviewer.WebViewer()

    faulthandler.enable()
    
    # add robot to the environments 
    env: LeggedRobot
    args.headless = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env._init_shared_memory(create_new=True, ready_event=None)
    # Then actions are generated from the planner. Make it list for transport across process

    # In this version ctrl_dt is equal to sim_dt 
    ctrl_dt = env_cfg.control.decimation * env_cfg.sim.dt 
    world_time = 0.0
    obs = env.get_observations()

    if args.web:
        web_viewer.setup(env)

    action_shm = shared_memory.SharedMemory(
        name="action_shm", create=False, size=env.num_envs * planner_cfg.planner.horizon * env.num_actions * 32
    )
    action_shared = np.ndarray(
        (env.num_envs, planner_cfg.planner.horizon, env.num_actions), dtype=np.float32, buffer=action_shm.buf
    )
    action_shared[:] = np.zeros((env.num_envs, planner_cfg.planner.horizon, env.num_actions), dtype=np.float32)

    plan_time_shm = shared_memory.SharedMemory(
            name="plan_time_shm", create=False, size=32
        )
    plan_time_shared = np.ndarray(
        1, dtype=np.float32, buffer=plan_time_shm.buf
    )
    plan_time_shared[0] = 10 # -ctrl_dt

    time_shm = shared_memory.SharedMemory(
                name="time_shm", create=False, size=32
            )
    time_shared = np.ndarray(1, dtype=np.float32, buffer=time_shm.buf)
    time_shared[0] = 0.0

    state_published = False
    # Enable interactive plotting
    # plt.ion()
    # fig, ax = plt.subplots()
    # line, = ax.plot([], [], label='Reward')
    # ax.set_xlim(0, 100)
    # ax.set_ylim(-10, 10)
    # ax.set_xlabel('Step')
    # ax.set_ylabel('Reward')
    # ax.set_title('Live Reward Plot')
    # ax.legend()
    # rewards = []
    # plt.show(block=False)

    # Simulate the environment 
    try:
        with torch.inference_mode():
            # for i in range(10*int(env.max_episode_length)):                      
            while True:
                # actions = planner.update(actions, priv_state.tolist(), extras)
                # now = time.time()
                actions_applied = torch.tensor(action_shared, dtype=torch.float32, device=env.device)
                while world_time <= (plan_time_shared[0] + ctrl_dt):
                    obs, priv_state, rews, done, infos = env.step(actions_applied[:, 0, :]) 
                    env.publish_planner_info()
                    # rewards.append(torch.mean(rews).item())
                    
                    # # Update plot data
                    # line.set_xdata(np.arange(len(rewards)))
                    # line.set_ydata(rewards)
                    
                    # # Adjust axis limits dynamically
                    # ax.set_xlim(0, len(rewards))
                    # ax.set_ylim(min(rewards)-1, max(rewards)+1)
                    # plt.draw()
                    # plt.pause(0.001)

                    world_time += ctrl_dt
                    time_shared[:] = world_time
                    # print("time is: ", world_time)
                    if not state_published:
                        ready_event.set()
                        state_published = True
                    if args.web:
                        web_viewer.render(fetch_results=True,
                                    step_graphics=True,
                                    render_all_camera_sensors=True,
                                    wait_for_page_load=True)

    except KeyboardInterrupt:
        print("Keyboard Terminated for Simulation")
    except Exception as e:
        # plt.ioff()
        # plt.show()
        print(f"Simulation Process Terminated: {e}")
    finally:
        env._close_shared_memory()
        action_shm.close()
        action_shm.unlink()
        time_shm.close()
        time_shm.unlink()
        plan_time_shm.close()
        plan_time_shm.unlink()
        print("finish clear the shared memory")

if __name__ == '__main__':
    args = get_args()
    
    # Load env configs 
    env_cfg, planner_cfg = task_registry.get_cfgs(name=args.task) # default is go2_stair
    env_cfg.env.num_samples_per_env = planner_cfg.planner.num_samples
    # We will need to rewrite the num_envs for initializing the planner environment
    env_cfg.env.num_agent_envs = env_cfg.env.num_envs

    ctrl_dt = env_cfg.control.decimation * env_cfg.sim.dt 
    action_shm = shared_memory.SharedMemory(
        name="action_shm", create=True, size=env_cfg.env.num_agent_envs * planner_cfg.planner.horizon * env_cfg.env.num_actions * 32
    )
    action_shared = np.ndarray(
        (env_cfg.env.num_agent_envs, planner_cfg.planner.horizon, env_cfg.env.num_actions), dtype=np.float32, buffer=action_shm.buf
    )
    action_shared[:] = np.zeros((env_cfg.env.num_agent_envs, planner_cfg.planner.horizon, env_cfg.env.num_actions), dtype=np.float32)

    plan_time_shm = shared_memory.SharedMemory(
            name="plan_time_shm", create=True, size=32
        )
    plan_time_shared = np.ndarray(
        1, dtype=np.float32, buffer=plan_time_shm.buf
    )
    plan_time_shared[0] = -ctrl_dt

    time_shm = shared_memory.SharedMemory(
                name="time_shm", create=True, size=32
            )
    time_shared = np.ndarray(1, dtype=np.float32, buffer=time_shm.buf)
    time_shared[0] = 0.0

    mp.set_start_method('spawn')
    ready_event = mp.Event()

    World(args, env_cfg, planner_cfg, ready_event)    
    # p1 = Process(target=Planner, args=(args, env_cfg, planner_cfg, ready_event))
    # p2 = Process(target=World, args=(args, env_cfg, planner_cfg, ready_event))
    # p1.start()
    # p2.start()
    # p1.join()
    # p2.join()
