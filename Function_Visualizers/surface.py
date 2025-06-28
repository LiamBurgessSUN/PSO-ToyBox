import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import sys
import os

# Add the parent directory to the Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.Testing.Loader import test_objective_function_classes as obj_functions
except ImportError:
    # Fallback: try importing from the current directory structure
    try:
        from ..SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.Testing.Loader import test_objective_function_classes as obj_functions
    except ImportError:
        print("Error: Could not import test_objective_function_classes. Please ensure you're running from the correct directory.")
        sys.exit(1)

function = obj_functions[1]()
name = function.__module__.split('.')[-1]

# 1. Create the grid of x and y values (same as before)
x = np.linspace(-100, 100, 100)
y = np.linspace(-100, 100, 100)
X, Y = np.meshgrid(x, y)

# 2. Define and evaluate the 3D function (same as before)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        Z[i, j] = function.evaluate(point)

# 3. Create the interactive 3D surface plot
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='plasma')])

# Update the layout with a title and axis labels
fig.update_layout(
    title=f'Interactive 3D Plot {name}',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    ),
    autosize=False,
    width=800,
    height=800,
    margin=dict(l=65, r=50, b=65, t=90)
)

# Show the interactive plot
fig.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none')

# Update the layout with a title and axis labels
ax.set_title(f'3D Plot of the {name} function')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
fig.colorbar(surf, shrink=0.5, aspect=5)

# Show the plot and keep it open
print(f"\nPlots generated for {name} function.")
print("Close the plot windows manually or press Ctrl+C to exit.")
plt.show(block=True)