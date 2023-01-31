
def create_string(euler_obj, pos_obj, size_obj, gravity, plane):
    if plane:
        plane_str = '<geom type="plane" pos="0 0 0" size="10 10 10" rgba="1 1 1 1"/>'
    else:
        plane_str = ""

    if gravity:
        gravity_str = ""
    else:
        gravity_str = '<option gravity="0 0 0"/>'
    return f"""
    <mujoco>
    {gravity_str}
    <worldbody>
        <light name="top" pos="0 0 1"/>
        <camera name="camera1" pos="1 -70 50" xyaxes="1 0 0 0 1 1.5"/>
        <body name="object_body" euler="{euler_obj}" pos="{pos_obj}">
            <joint name="joint1" type="free"/>
            <geom name="object_geom" type="box" size="{size_obj}" rgba="1 0 0 1"/>
        </body>
        {plane_str}
    </worldbody>
    </mujoco>
    """

    # if plane and gravity:
    #     return f"""
    #     <mujoco>
    #     <worldbody>
    #         <light name="top" pos="0 0 1"/>
    #         <camera name="camera1" pos="1 -70 50" xyaxes="1 0 0 0 1 1.5"/>
    #         <geom type="plane" pos="0 0 0" size="10 10 10" rgba="1 1 1 1"/>
    #         <body name="object_body" euler="{euler_obj}" pos="{pos_obj}">
    #             <joint name="joint1" type="free"/>
    #             <geom name="object_geom" type="box" size="{size_obj}" rgba="1 0 0 1"/>
    #         </body>
    #     </worldbody>
    #     </mujoco>
    #     """
    # elif plane and not gravity:
    #     return f"""
    #     <mujoco>
    #     <option gravity="0 0 0"/>
    #     <worldbody>
    #         <light name="top" pos="0 0 1"/>
    #         <camera name="camera1" pos="1 -70 50" xyaxes="1 0 0 0 1 1.5"/>
    #         <geom type="plane" pos="0 0 0" size="10 10 10" rgba="1 1 1 1"/>
    #         <body name="object_body" euler="{euler_obj}" pos="{pos_obj}">
    #             <joint name="joint1" type="free"/>
    #             <geom name="object_geom" type="box" size="{size_obj}" rgba="1 0 0 1"/>
    #         </body>
    #     </worldbody>
    #     </mujoco>
    #     """
    # elif gravity and not plane:
    #     return f"""
    #     <mujoco>
    #     <worldbody>
    #         <light name="top" pos="0 0 1"/>
    #         <camera name="camera1" pos="1 -70 50" xyaxes="1 0 0 0 1 1.5"/>
    #         <body name="object_body" euler="{euler_obj}" pos="{pos_obj}">
    #             <joint name="joint1" type="free"/>
    #             <geom name="object_geom" type="box" size="{size_obj}" rgba="1 0 0 1"/>
    #         </body>
    #     </worldbody>
    #     </mujoco>
    #     """
    # elif not gravity and not plane:
    #     return f"""
    #     <mujoco>
    #     <option gravity="0 0 0"/>
    #     <worldbody>
    #         <light name="top" pos="0 0 1"/>
    #         <camera name="camera1" pos="1 -70 50" xyaxes="1 0 0 0 1 1.5"/>
    #         <body name="object_body" euler="{euler_obj}" pos="{pos_obj}">
    #             <joint name="joint1" type="free"/>
    #             <geom name="object_geom" type="box" size="{size_obj}" rgba="1 0 0 1"/>
    #         </body>
    #     </worldbody>
    #     </mujoco>
    #     """




# def create_string():
#     return """<mujoco>
#   <worldbody>
#     <body name="box" pos="0 0 2">
#       <geom type="box" size=".2 .2 .2"/>
#       <inertial pos=".1 .1 .1" mass="1" diaginertia="1 1 1"/>
#       <joint type="free" damping="1" armature="1"/>
#       <body>
#         <geom type="sphere" size=".1" pos=".1 .1 .1"/>
#       </body>
#     </body>
#   </worldbody>
# </mujoco>
# """
