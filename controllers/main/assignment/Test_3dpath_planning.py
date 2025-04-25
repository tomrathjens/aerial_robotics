import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_targets(filename):
    data = np.loadtxt(filename, delimiter=',')
    positions = data[:, :3]
    yaws = data[:, 3]
    return positions, yaws

def catmull_rom_segment(p0, p1, p2, p3, num_points=20):
    points = []
    for t in np.linspace(0, 1, num_points):
        t2 = t * t
        t3 = t2 * t
        point = 0.5 * (
            (2 * p1) +
            (-p0 + p2) * t +
            (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
            (-p0 + 3*p1 - 3*p2 + p3) * t3
        )
        points.append(point)
    return np.array(points)

def generate_path(positions, num_points=20):
    padded = np.vstack([
        positions[0] - (positions[1] - positions[0]),
        positions,
        positions[-1] + (positions[-1] - positions[-2])
    ])
    curve = []
    for i in range(len(padded) - 3):
        segment = catmull_rom_segment(
            padded[i], padded[i+1], padded[i+2], padded[i+3],
            num_points=num_points
        )
        curve.append(segment)
    return np.vstack(curve)

def compute_yaws_from_curve(curve):
    deltas = np.diff(curve, axis=0)
    yaws = np.arctan2(deltas[:, 1], deltas[:, 0])
    yaws = np.append(yaws, yaws[-1])
    return yaws

positions, yaws_initial = load_targets("controllers/main/assignment/gate_coordinates2backup.txt")
curve = generate_path(positions, num_points=20)
yaws_curve = compute_yaws_from_curve(curve)
curve_with_yaw = np.hstack([curve, yaws_curve[:, np.newaxis]])
print(curve_with_yaw)

print("Size of curve:", curve_with_yaw.shape)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'b-', label='Spline Path', linewidth=2)
ax.scatter(*positions.T, color='red', s=80, label='Waypoints')

for i, pos in enumerate(positions):
    ax.text(pos[0], pos[1], pos[2], f' {i}', color='black', fontsize=12, weight='bold')

arrow_length = 0.5
for i, pos in enumerate(positions):
    dx = arrow_length * np.cos(yaws_initial[i])
    dy = arrow_length * np.sin(yaws_initial[i])
    dz = 0
    ax.quiver(pos[0], pos[1], pos[2], dx, dy, dz, color='green', length=arrow_length)

arrow_spacing = 20
for i in range(0, len(curve), arrow_spacing):
    dx = 0.5 * np.cos(yaws_curve[i])
    dy = 0.5 * np.sin(yaws_curve[i])
    dz = 0
    ax.quiver(curve[i, 0], curve[i, 1], curve[i, 2], dx, dy, dz, color='orange', length=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Path with Catmull-Rom Splines and Yaw')
ax.legend()
plt.grid(True)
plt.show()
