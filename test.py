import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt

from simulation.environment import atmosphere

geometric_altitudes = jnp.linspace(0, 81e3, 10000)

pressure = atmosphere.pressure(geometric_altitudes)
temperature = atmosphere.temperature(geometric_altitudes)
grav_acceleration = atmosphere.grav_acceleration(geometric_altitudes)
thermal_conductivity = atmosphere.thermal_conductivity(geometric_altitudes)
density = atmosphere.density(geometric_altitudes)

# Plotting
# Use interactive backend for matplotlib
matplotlib.use("QtAgg")

# fig, ax1 = plt.subplots()

# color = "tab:red"
# ax1.set_xlabel("Pressure [Pa]")
# ax1.set_ylabel("Geometric altitude [km]")
# ax1.plot(pressure, geometric_altitudes / 1e3, color=color)
# ax1.grid()

# ax2 = ax1.twiny()
# color = "tab:blue"
# ax2.set_xlabel("Temperature [K]")
# ax2.plot(temperature, geometric_altitudes / 1e3, color=color)
# ax2.grid()

# fig.tight_layout()

# plt.show()

# Plot Density
plt.figure()
plt.plot(density, geometric_altitudes / 1e3)
plt.xlabel("Density [kg/m^3]")
plt.ylabel("Geometric altitude [km]")
plt.grid()
plt.show()
