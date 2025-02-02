


import jax
from jax import config
import jax.numpy as np
config.update("jax_enable_x64", True)

from DMA_MR_ss import *
import matplotlib.pyplot as plt
from opyrability import implicit_map



theta = np.linspace(0, 2 * np.pi, 400)
phi = np.pi / 4
a, b= 0.15, 1
h, k = 22.4 , 39.4
y1 = h +  (a * np.cos(theta) * np.cos(phi) - b * np.sin(theta) * np.sin(phi))  
y2 = k +  (b * np.sin(theta) * np.cos(phi) + a * np.cos(theta) * np.cos(phi))



AIS_PTS=np.array([y1,y2]).T
plt.plot(AIS_PTS[:,0], AIS_PTS[:,1])
output_init = np.array([480.00, 600.00])




AIS, AOS, AIS_poly, AOS_poly = implicit_map(dma_mr_uncertain_flows,  
                                        output_init,
                                        continuation='odeint',
                                        domain_points=AIS_PTS,
                                        direction = 'inverse')





AOS_PTS = AOS.reshape(-1,2)
plt.figure()
plt.plot(AOS_PTS[1:,0], AOS_PTS[1:,1])





from jax import random
# Set the key for random number generation
key = random.PRNGKey(0)

# Number of simulation points and center of ellipse.
num_simulations = 10000
a, b= 0.15, 1


# Scaling factor for 95% confidence interval in 2D
from scipy.stats import chi2
alpha = 0.95 # Confidence
dof   = 2    # Degrees of freedom
# scaling factor, 2.4477 for 95%
scaling_factor  = np.sqrt(chi2.ppf(alpha, dof))



# Adjust the a and b values - here a and b are adjusted to be able to build the covariance matrix to draw the multivariate normal distribution that will be within 95% of the cloud of points.
a_adjusted = a / scaling_factor
b_adjusted = b / scaling_factor


# Constructing the covariance matrix using a, b, and rotation matrix

covariance_matrix_initial = np.array([[a_adjusted**2,  0], 
                                      [0,              b_adjusted**2]])


rotation_matrix = np.array([[np.cos(phi), -np.sin(phi)], 
                            [np.sin(phi), np.cos(phi)]])


covariance_matrix_constructed = rotation_matrix @ covariance_matrix_initial @ rotation_matrix.T


# Plotting the ellipses with constructed covariance matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(y1, y2, 'r--', label='Ellipse from Parametric Equations')
# Monte Carlo Sampling simulations
mean = np.array([h, k])
samples = random.multivariate_normal(key, mean, covariance_matrix_constructed, (num_simulations,))

hb=ax.hexbin(samples[:, 0], samples[:, 1], gridsize=150, cmap='viridis', bins='log')
fig.colorbar(hb, ax=ax, label='log10(N)')
ax.set_xlabel('Benzene production [mg/h]')
ax.set_ylabel('Natural gas conversion [\%]')
ax.set_title('Comparison vs Monte Carlo Sampling')
ax.legend()
plt.show()





from opyrability import nlp_based_approach
from DMA_MR_ss import dma_mr_uncertain_flows_check
benzene_samples = samples[:, 0]
ch4_samples = samples[:, 1]


# Calculating for each Monte Carlo simulation
v0_values = []
v_He_values = []
DOS_resolution = [1, 1]
u0 = output_init
lb = np.array([300, 300])
ub = np.array([1000, 1000])
for i in range(num_simulations):
    benzene = benzene_samples[i]
    ch4 = ch4_samples[i]
    
    DOS_bounds = np.array([[benzene, benzene],
                          [ch4, ch4]])

    fDIS, fDOS, conv = nlp_based_approach(dma_mr_uncertain_flows_check,
                                              DOS_bounds, 
                                              DOS_resolution,
                                              u0, 
                                              lb,ub,
                                              method='ipopt', 
                                              plot=False,
                                              ad=False,
                                              warmstart=True)
    
    v0_values.append(fDIS[:,0])
    v_He_values.append(fDIS[:,1])



# Adjusting global font size
plt.rcParams.update({'font.size': 14})

fig, ax1 = plt.subplots(figsize=(7,6))
hb = ax1.hexbin(benzene_samples, ch4_samples, gridsize=150, cmap='viridis', bins='log')
fig.colorbar(hb, ax=ax1, label='log10(N)')
ax1.plot(y1, y2, 'r--', label='Disturbance region described by closed-path')
ax1.set_title('Output variables s.t disturbances')
ax1.set_xlabel('Benzene production (mg/h)')
ax1.set_ylabel('Natural Gas conversion (%)')
fig.tight_layout()
plt.show()



fig, ax2 = plt.subplots(figsize=(7,6))
hb = ax2.hexbin(v0_values, v_He_values, gridsize=150, cmap='viridis', bins='log')
fig.colorbar(hb, ax=ax2, label='log10(N)')
ax2.plot(AOS_PTS[1:,0], AOS_PTS[1:,1],'r--', label='opyrability')
ax2.set_title('Input variables s.t disturbances')
ax2.set_xlabel('Tube flow rate [cm3/h]')
ax2.set_ylabel('Shell flow rate [cm3/h]')
fig.tight_layout()
plt.show()


import numpy as npp
v_He_hexbin = npp.array(v_He_values)
v0_hexbin = npp.array(v0_values)
AOS_PTS_hexbin = npp.array(AOS_PTS)


npp.savez('montecarlo_data.npz', v_He_hexbin, v0_hexbin, AOS_PTS_hexbin)