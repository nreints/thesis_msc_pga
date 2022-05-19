#Importing OpenAI gym package and MuJoCo engine
import gym
import mujoco_py
import numpy as np
import os

# #Setting MountainCar-v0 as the environment
# env = gym.make('MountainCar-v0')
# env = gym.make('Ant-v3')
# model = env.model

# #Sets an initial state
# env.reset()

# # Get xml from model
# # env.model.get_xml()

# # Get quaternion velocities of the joints
# # env.data.qvel

# # Get quaternion positions of the joints
# # env.data.qpos

# # Get number of generalized coordinates = dim(qpos) \neq number of joints
# # model.nq

# print("'1'",env.model.nq, env.model.nv, env.model.njnt)
# print(env.model.get_xml())
# print(env.model.body_names)
# for i in range(len(env.model.body_names)):
#     print(i, env.model.body_names[i], env.model.body_parentid[i], env.model.body_pos[i])

# # Rendering our instance 300 times
# # for _ in range(500):
# #   #renders the environment
# #   env.render()
# #   #Takes a random action from its action space
# #   # aka the number of unique actions an agent can perform
# # #   print("HOPE", env.state_vector())
# #   # print(env.get_xml())
# #   # print(env.save())
# #   # observation, reward, done, info = env.step(env.action_space.sample())
# #   env.step(env.action_space.sample())
# env.close()

#########################################

MODEL_XML_pyramid =  """
<?xml version="1.0" ?>
<mujoco>
   <asset>
      <mesh name="pyramid" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
   </asset>
   <worldbody>
      <light diffuse=".5 .5 .5" pos="0 0 0" dir="0 0 -1"/>
      <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
      <body name="pyr1" pos="0 0 2">
         <joint name="joint" type="free"/>
         <geom type="mesh" mesh="pyramid"/>
      </body>
      <camera fovy="180" mode="fixed" name="camera1" pos="0.0 0.0 0.0" euler="0.2 1.2 1.57"/>
   </worldbody>
</mujoco>"""

MODEL_XML_blokkie =  """
<?xml version="1.0" ?>
<mujoco>
   <asset>
      <mesh name="blokkie" vertex="0 0 0  1 0 0  0 1 0  1 1 0  0 0 1  1 0 1  0 1 1  1 1 1"/>
   </asset>
   <worldbody>
      <light diffuse=".5 .5 .5" pos="0 0 0" dir="0 0 -1"/>
      <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
      <body name="blokje1" pos="0 0 50">
         <geom type="mesh" mesh="blokkie"/>
         <joint name="joint1" type="free"/>
      </body>
      <camera fovy="180" mode="fixed" name="camera1" pos="0.0 0.0 0.0" euler="0.2 1.2 1.57"/>
   </worldbody>
</mujoco>"""

# mj_path = mujoco_py.utils.discover_mujoco()
# # mj_path = /Users/nienkereints/.mujoco/mujoco210/model/

# xml_path = os.path.join(mj_path, 'model', 'blokkie.xml')

# model = mujoco_py.load_model_from_path(xml_path)
# model = mujoco_py.load_model_from_xml(MODEL_XML_pyramid)
model = mujoco_py.load_model_from_xml(MODEL_XML_blokkie)

sim = mujoco_py.MjSim(model)
# viewer = mujoco_py.MjViewer(sim)
# viewer.render()
sim.reset()
print(type(sim.data))
print(model.nq)
print(sim.data.qpos)
print(sim.data.body_xquat)

def rotVecQuat(v, q):
    res = np.zeros(3)
    mujoco_py.functions.mju_rotVecQuat(res, v, q)
    return res

# print(model.body_names)

# viewer = mujoco_py.MjViewer(sim)
# print("----")
# print(sim.named.data.qpos)
# print(sim.data.xaxis)
# # print(sim.data.qpos)
# # print(model.nq)

# for i in range(1000):
#    #  print(sim.get_state())
#    #  print("vert\t", model.mesh_vert)
#    if i % 100 == 0:
#       print("----- iter ", i, "-----")
#       print("qpos\t", sim.data.qpos)
      # print("pos\t", sim.data.xanchor)
      # print("axis\t", sim.data.xaxis)
#       #  print(sim.data.qpos)
#    sim.step()
#    viewer.render()

# sim.reset()




# Get xml from model
# env.model.get_xml()

# Get quaternion velocities
# sim.data.qvel

# Get quaternion positions
# sim.data.qpos

# Get number of generalized coordinates = dim(qpos) \neq number of joints
# model.nq

# mjtNum*   xanchor;              // Cartesian position of joint anchor       (njnt x 3)
# mjtNum*   xaxis;                // Cartesian joint axis                     (njnt x 3)


# # print(sim.data.state_vector)
# sim.step()
# print(sim.data.qpos)

