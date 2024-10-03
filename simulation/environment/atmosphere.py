"""
Compute atmospheric properties for heights ranging from -5 km to 80 km.

The implementation is based on the ICAO standard atmosphere from 1993.

References:
===========

.. [ICAO93] International Civil Aviation Organization ; Manual Of The ICAO
            Standard Atmosphere -- 3rd Edition 1993 (Doc 7488) -- extended
            to 80 kilometres (262 500 feet)

.. [WISA19] Wikipedia ; International Standard Atmosphere ;
            https://en.wikipedia.org/wiki/International_Standard_Atmosphere

.. [AMBI22] Ambiance Python Package ;
            https://github.com/airinnova/ambiance/blob/master/src/ambiance/ambiance.py#L347
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

##################### CONSTANTS #####################
# Primary constants (table A)
g_0 = 9.80665  # m/s^2
M_0 = 28.964420e-3
N_A = 602.257e21
P_0 = 101.325e3
R_star = 8314.32e-3
R = 287.05287
S = 110.4
T_i = 273.15
T_0 = 288.15
t_i = 0.0
t_0 = 15.0
beta_s = 1.458e-6
kappa = 1.4
rho_0 = 1.225
sigma = 0.365e-9

# Additional constants
r_0 = 6_356_766

# Geometric heights
h_min = -5_004
h_max = 81_020

# Geopotential heights
H_min = -5_000
H_max = 80_000

# Layers
layer_names = [
    "Troposphere",
    "Troposphere",
    "Tropopause",
    "Stratosphere",
    "Stratosphere",
    "Stratopause",
    "Mesosphere",
    "Mesosphere",
    "Mesosphere",
]

layer_base_geopotential_altitudes = jnp.array(
    [-5.0e3, 0.0e3, 11.0e3, 20.0e3, 32.0e3, 47.0e3, 51.0e3, 71.0e3, 80.0e3]
)

layer_base_temperatures = jnp.array(
    [320.65, 288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 196.65]
)

layer_base_temperature_gradients = jnp.array(
    [-6.5e-3, -6.5e-3, 0.0e-3, 1.0e-3, 2.8e-3, 0.0e-3, -2.8e-3, -2.0e-3, -2.0e-3]
)

layer_base_pressures = jnp.array(
    [
        1.77687e5,
        1.01325e5,
        2.26320e4,
        5.47487e3,
        8.68014e2,
        1.10906e2,
        6.69384e1,
        3.95639e0,
        8.86272e-1,
    ]
)


def geometric_to_geopotential_altitude(
    z: Float[Array, "*dims"],
) -> Float[Array, "*dims"]:
    return z * r_0 / (r_0 + z)


def geopotential_to_geometric_altitude(
    H: Float[Array, "*dims"],
) -> Float[Array, "*dims"]:
    return H * r_0 / (r_0 - H)


def layer_num(H: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    return jnp.searchsorted(layer_base_geopotential_altitudes, H, side="right") - 1


def layer_properties(
    H: Float[Array, "*dims"],
) -> tuple[
    Float[Array, "*dims"],
    Float[Array, "*dims"],
    Float[Array, "*dims"],
    Float[Array, "*dims"],
]:
    n = layer_num(H)
    H_n = layer_base_geopotential_altitudes[n]
    T_n = layer_base_temperatures[n]
    L_n = layer_base_temperature_gradients[n]
    P_n = layer_base_pressures[n]

    return H_n, T_n, L_n, P_n


def pressure(z: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    H = geometric_to_geopotential_altitude(z)
    H_n, T_n, L_n, P_n = layer_properties(H)

    return jnp.where(
        L_n == 0,
        P_n * jnp.exp(-g_0 * (H - H_n) / (R * T_n)),
        P_n
        * (1 + (L_n / T_n) * (H - H_n))
        ** (jnp.where(L_n == 0, 0, 1 / L_n) * (-g_0 / R)),
    )


def temperature(z: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    H = geometric_to_geopotential_altitude(z)
    H_n, T_n, L_n, _ = layer_properties(H)

    return T_n + L_n * (H - H_n)


def grav_acceleration(z: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    return g_0 * (r_0 / (r_0 + z)) ** 2


def collision_frequency(z: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    return (
        4
        * sigma**2
        * N_A
        * jnp.sqrt(jnp.pi / (R * M_0))
        * pressure(z)
        / jnp.sqrt(temperature(z))
    )


def density(z: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    return pressure(z) / (R * temperature(z))


def dynamic_viscosity(z: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    T = temperature(z)
    return beta_s * T**1.5 / (T + S)


def kinematic_viscosity(z: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    return dynamic_viscosity(z) / density(z)


def number_density(z: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    return N_A * pressure(z) / (R * temperature(z))


def mean_free_path(z: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    return 1 / (jnp.sqrt(2) * jnp.pi * sigma**2 * number_density(z))


def mean_particle_speed(z: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    return jnp.sqrt((8 / jnp.pi) * R * temperature(z))


def pressure_scale_height(z: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    return R * temperature(z) / grav_acceleration(z)


def specific_weight(z: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    return density(z) * grav_acceleration(z)


def speed_of_sound(z: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    return jnp.sqrt(kappa * R * temperature(z))


def thermal_conductivity(z: Float[Array, "*dims"]) -> Float[Array, "*dims"]:
    T = temperature(z)
    return 2.648151e-3 * T**1.5 / (T + 245.4 * 10 ** (-12 / T))


if __name__ == "__main__":
    z = jnp.array([0.0, 15e3, 30e3, 45e3, 60e3, 75e3])
    H = geometric_to_geopotential_altitude(z)
    n = layer_num(H)
    P = pressure(z)
    print(P)
    print(n)
