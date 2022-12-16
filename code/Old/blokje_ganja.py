d = 3

# Create a hypercube.
points = [
    ('0000' + bin(i)[2:]).zfill(d)
    for i in range(2 ** d)
]
points = [
    sum(
        [x - 0.5 for x in map(int, list(x))] * [1e1, 1e2, 1e3, 1e4]
    )
    for x in points
]

# The edges of the hypercube.
lines = [
    [points[i], points[j]]
    for i in range(len(points))
    for j in range(len(points))
    if (i < j or (i ^ j) & (i ^ j - 1))
]

# The attachment point of the spring.
attach = points[-1]

# The Forques
def F(M, B):
    Gravity = -9.81e2 * M
    Damping = -0.5 * B**2
    Hooke = 16 * (attach - M)
    return Gravity + Damping + Hooke

# The physics state : current pos/rot and current lin/ang velocity.
State = [1, 1e12 + 2e13 - 1e24]

# The differential of the state
def dS(M, B):
    return [-0.5 * M * B, F(M, B) - 0.5 * (B**2 - B)]

# Update the state
for i in range(10):
    State = State + 1/600 * dS(*State)


# Render
# code for rendering goes here