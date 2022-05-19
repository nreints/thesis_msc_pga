import mujoco_py
import os

mj_path = mujoco_py.utils.discover_mujoco()
# mj_path = /Users/nienkereints/.mujoco/mujoco210/model/

xml_path = os.path.join(mj_path, 'model', 'blokkie.xml')

model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)
print("this", model.nq) 
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# print(sim.data.state_vector)
sim.mj_printData()
sim.step()
print(sim.data.qpos)