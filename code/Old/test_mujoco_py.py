#Importing OpenAI gym package and MuJoCo engine
import gym
import mujoco_py
#Setting MountainCar-v0 as the environment
env = gym.make('MountainCar-v0')
env = gym.make('Ant-v3')
# env = gym.make('MsPacman-v0')

#Sets an initial state
env.reset()
# print(env.state_vector())
print(len(env.data.qvel))
fioeh
print(env.data.qpos[0])
print(env.data.qvel.flat[0])
# Rendering our instance 300 times
for _ in range(500):
  #renders the environment
  env.render()
  #Takes a random action from its action space
  # aka the number of unique actions an agent can perform
  print("HOPE", env.state_vector())
  # print(env.get_xml())
  # print(env.save())
  # observation, reward, done, info = env.step(env.action_space.sample())
  env.step(env.action_space.sample())

  # print(observation)
env.close()

