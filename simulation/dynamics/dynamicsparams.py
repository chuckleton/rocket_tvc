import dataclasses

import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclasses.dataclass
class DynamicsParams:
    m_B: Float[Array, "1 1"]
    J_A: Float[Array, "3 3"]
    J_B: Float[Array, "3 3"]
    C_B: Float[Array, "3 2"]
    rho_A_joint: Float[
        Array, "3 1"
    ]  # Position vector of the joint relative to body A COM
    rho_joint_B: Float[
        Array, "3 1"
    ]  # Position vector of the body B COM relative to the joint


m_A_0 = 25.0
m_B = 7.0
J_A = jnp.eye(3) * 1.0
J_B_0 = jnp.eye(3) * 0.1
C_B = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
rho_A_joint = jnp.array([0.0, 0.0, -1.0]).reshape(-1, 1)
rho_joint_B = jnp.array([0.0, 0.0, 0.25]).reshape(-1, 1)

x_A_0 = jnp.array([0.0, 0.0, 0.0]).T
mrp_A_0 = jnp.array([0.0, 0.0, 0.0]).T
r_0 = jnp.array([0.0, 0.0]).T
v_A_0 = jnp.array([0.0, 0.0, 10.0]).T
omega_A_0 = jnp.array([0.0, 0.0, 0.0]).T
r_dot_0 = jnp.array([0.0, 0.0]).T

params = DynamicsParams(m_B, J_A, J_B_0, C_B, rho_A_joint, rho_joint_B)

x_0 = jnp.concatenate(
    [
        x_A_0,
        mrp_A_0,
        r_0,
        v_A_0,
        omega_A_0,
        r_dot_0,
        jnp.array([m_A_0]),
        jnp.array([0.0]),
    ]
).reshape(-1, 1)
