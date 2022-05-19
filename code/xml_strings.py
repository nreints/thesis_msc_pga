model_xml_swinging_body = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model_xml_body_swinging_body = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
    </body>
  </worldbody>
</mujoco>
"""

model_xml_blokje = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body name="box" euler="0 0 -80" pos="0 0 10">
      <joint name="joint1" type="free"/>
      <geom name="blokje" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    </body>
    <camera fovy="90" name="camera1" mode="targetbodycom" target="box" quat="0.442 0 0.0 0"/>
    <camera fovy="90" mode="fixed" name="camera2" pos="0.0 0.0 5.0" />
  </worldbody>
</mujoco>
"""


model_xml_pyramid =  """
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

model_xml_blokje_mesh =  """
<?xml version="1.0" ?>
<mujoco>
   <asset>
      <mesh name="blokje" vertex="0 0 0  1 0 0  0 1 0  1 1 0  0 0 1  1 0 1  0 1 1  1 1 1"/>
   </asset>
   <worldbody>
      <light diffuse=".5 .5 .5" pos="0 0 0" dir="0 0 -1"/>
      <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
      <body name="blokje1" pos="0 0 50">
         <geom type="mesh" mesh="blokkie"/>
         <joint name="joint1" pos="0 0 -50" type="free"/>
      </body>
      <camera fovy="180" mode="fixed" name="camera1" pos="0.0 0.0 0.0" euler="0.2 1.2 1.57"/>
   </worldbody>
</mujoco>"""