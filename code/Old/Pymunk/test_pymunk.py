import pymunk               # Import pymunk..

import platform
mac_ver_float = float('.'.join(platform.mac_ver()[0].split('.')[:2]))
if mac_ver_float > 10.12:
    compiler_preargs += ['-arch', 'x86_64']
else:
    compiler_preargs += ['-arch', 'i386', '-arch', 'x86_64']

space = pymunk.Space()      # Create a Space which contain the simulation
space.gravity = 0,-981      # Set its gravity

body = pymunk.Body()        # Create a Body
body.position = 50,100      # Set the position of the body

poly = pymunk.Poly.create_box(body) # Create a box shape and attach to body
poly.mass = 10              # Set the mass on the shape
space.add(body, poly)       # Add both body and shape to the simulation

print_options = pymunk.SpaceDebugDrawOptions() # For easy printing

while True:                 # Infinite loop simulation
    space.step(0.02)        # Step the simulation one step forward
    space.debug_draw(print_options) # Print the state of the simulation