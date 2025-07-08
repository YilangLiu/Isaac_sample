# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""Trains a hopper to run in the +x direction."""

from typing import Tuple

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from brax import envs
import jax
from jax import numpy as jp
import time 


# instantiate the environment
env_name = 'hopper'
env = envs.get_environment(env_name, backend="mjx")

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(jax.vmap(env.step, in_axes=(None, 0), out_axes=(0)))

state = jit_reset(jax.random.PRNGKey(0))
num_envs = 4096
nstep = 1
# grab a trajectory

for i in range(nstep):
  ctrl = -0.1 * jp.ones((num_envs, env.sys.nu))
  jit_step(state, ctrl)
now = time.time()
for i in range(nstep):
  ctrl = -0.1 * jp.ones((num_envs, env.sys.nu))
  jit_step(state, ctrl)
print("time is ", time.time() - now)