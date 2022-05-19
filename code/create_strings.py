
def create_string(euler_obj, pos_obj, type_obj, size_obj):
    return f"""
    <mujoco>
    <worldbody>
        <light name="top" pos="0 0 1"/>
        <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
        <body name="object_body" euler="{euler_obj}" pos="{pos_obj}">
        <joint name="joint1" type="free"/>
        <geom name="object_geom" type="{type_obj}" size="{size_obj}" rgba="1 0 0 1"/>
        </body>
    </worldbody>
    </mujoco>
    """

# print(create_string("a", "b", "c", "d"))