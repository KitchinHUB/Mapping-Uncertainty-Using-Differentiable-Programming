import jax.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from jax import random

plt.rcParams.update({'font.size': 16})

# Load data that was generated in the Jupyter notebook session.
data = np.load('montecarlo.npz')
v_He_values = data['arr_0']
v0_values = data['arr_1']
DIS_points = data['arr_2']

# Generate ellipse data
theta = np.linspace(0, 2 * np.pi, 400)
phi = np.pi / 4
a, b = 0.15, 1
h, k = 22.4, 39.4
y1 = h + (a * np.cos(theta) * np.cos(phi) - b * np.sin(theta) * np.sin(phi))
y2 = k + (b * np.sin(theta) * np.cos(phi) + a * np.cos(theta) * np.cos(phi))

# Set the key for random number generation
key = random.PRNGKey(0)

# Number of simulation points and center of ellipse
num_simulations = 10000

# Scaling factor for 95% confidence interval in 2D
alpha = 0.95  # Confidence
dof = 2  # Degrees of freedom
scaling_factor = np.sqrt(chi2.ppf(alpha, dof))  # Take the square root

# Adjust the a and b values
a_adjusted = a / scaling_factor
b_adjusted = b / scaling_factor

# Construct covariance matrix
covariance_matrix_initial = np.array([[a_adjusted**2, 0],
                                       [0, b_adjusted**2]])
rotation_matrix = np.array([[np.cos(phi), -np.sin(phi)],
                            [np.sin(phi), np.cos(phi)]])
covariance_matrix_constructed = rotation_matrix @ covariance_matrix_initial @ rotation_matrix.T

# Generate Monte Carlo samples
mean = np.array([h, k])
samples = random.multivariate_normal(key, mean, covariance_matrix_constructed, (num_simulations,))

# Create side-by-side subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 8))

# First subplot: Output Space Variables
axs[0].plot(y1, y2, 'r--', label='Ellipse from Parametric Equation', linewidth=3)
hb1 = axs[0].hexbin(samples[:, 0], samples[:, 1], gridsize=50, cmap='viridis', bins='log', label='Monte Carlo (Hexagonal bins)')
cb1 = fig.colorbar(hb1, ax=axs[0], label='$log_{10}(N)$')
axs[0].set_xlabel('Benzene ($C_{6}H_{6}$) production [mg/h]')
axs[0].set_ylabel('Methane ($CH_4$) conversion [%]')
axs[0].set_title('Output Space Variables (Domain - AOS)')
axs[0].legend()

# Second subplot: Input Space Variables
axs[1].plot(DIS_points[1:, 0], DIS_points[1:, 1], 'r--', label='Uncertainty Mapping', linewidth=3)
hb2 = axs[1].hexbin(v0_values, v_He_values, gridsize=50, cmap='viridis', bins='log', label='Monte Carlo (Hexagonal bins)')
cb2 = fig.colorbar(hb2, ax=axs[1], label='$log_{10}(N)$')
axs[1].set_xlabel('Tube flow rate [$cm^3/h$]')
axs[1].set_ylabel('Shell flow rate [$cm^3/h$]')
axs[1].set_title('Input Space Variables (Image - AIS)')
axs[1].legend()

# Adjust layout and save as PDF
plt.tight_layout()
# plt.savefig('implicit_inverse_dma.pdf', format='pdf')
plt.show()


import matplotlib.pyplot as plt
import matplotlib.path as mpath
import seaborn as sns
import numpy as np

# Define the path and points
path = mpath.Path(np.array(DIS_points))
points = np.hstack([v0_values, v_He_values])

# Calculate inside and outside points
inside_points = path.contains_points(points)
points_in = np.sum(inside_points)
points_out = len(points) - points_in

# Print results
print('There are', points_in, 'points within the ellipse')
print('There are', points_out, 'points outside the ellipse')
print(f'{(points_in / num_simulations) * 100:.2f}% of points are inside the ellipse')

# Hexagonal bin plot with seaborn
sns.set_theme(style="ticks")  # Use style without a grid
g = sns.jointplot(
    x=v0_values.reshape(-1), 
    y=v_He_values.reshape(-1), 
    kind="hex", 
    color="#002855"
)
g.fig.set_size_inches(10, 8)

# Increase font size
font_size = 18
g.ax_joint.set_xlabel('Tube flow rate [$cm^3/h$]', fontsize=font_size)
g.ax_joint.set_ylabel('Shell flow rate [$cm^3/h$]', fontsize=font_size)
g.fig.suptitle(
    'Hexagonal Bin - Input Variables with Disturbances', 
    fontsize=font_size + 2, 
    y=1.03
)

# Overlay the uncertainty mapping curve
line = g.ax_joint.plot(
    DIS_points[1:, 0], 
    DIS_points[1:, 1], 
    'r--', 
    label="Uncertainty Mapping"
)

# Remove gridlines
g.ax_joint.grid(False)

# Increase tick label font size
g.ax_joint.tick_params(axis='both', which='major', labelsize=font_size - 2)

# Add legend with larger font size
g.ax_joint.legend(handles=line, loc='upper right', fontsize=font_size - 4)

# Save the plot as a PDF with adjusted margins
g.fig.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout without cropping the title
# g.fig.savefig('hexagonal_input_disturbances.pdf', format='pdf')

plt.show()