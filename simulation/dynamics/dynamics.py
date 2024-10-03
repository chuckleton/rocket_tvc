import dataclasses

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def mrp2rotm(mrp: Float[Array, "3 1"]) -> Float[Array, "3 3"]:
    """Converts an MRP vector (3x1) to a rotation matrix (3x3).

    Taken from https://github.com/mlourakis/MRPs/blob/master/mrp2rot.m

    Returns:
        Float[Array, "3 3"]: The rotation matrix.
    """
    sqmag = jnp.dot(mrp, mrp)
    skw = skew(mrp)
    R = (
        jnp.eye(3) + ((4 * (1 - sqmag) * jnp.eye(3) + 8 * skw) * skw) / (1 + sqmag) ** 2
    )  # (255b) from Shuster

    return R


def rotm2mrp(R: Float[Array, "3 3"]) -> Float[Array, "3 1"]:
    """Converts a rotation matrix (3x3) to an MRP vector (3x1).

    Taken from https://github.com/mlourakis/MRPs/blob/master/rot2mrp.m
    Note: might return the shadow MRP -m/(m*m') which represents the same rotation as m

    Returns:
        Float[Array, "3 1"]: The MRP vector.
    """
    rho = (
        -1
        / (1 + jnp.trace(R))
        * jnp.array([R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0]]).T
    )  # Rodrigues vector, see (204) in Shuster; seems to have a wrong sign
    mrp = rho / (1 + jnp.sqrt(1 + jnp.dot(rho, rho)))  # (3.138) in Schaub & Junkins

    return mrp


@dataclasses.dataclass
class DynamicsParams:
    m_B: Float[Array, "1 1"]
    J_A: Float[Array, "3 3"]
    J_B: Float[Array, "3 3"]
    rho_A_joint: Float[
        Array, "3 1"
    ]  # Position vector of the joint relative to body A COM
    rho_joint_B: Float[
        Array, "3 1"
    ]  # Position vector of the body B COM relative to the joint


def R_joint(r: Float[Array, "2 1"]) -> Float[Array, "3 3"]:
    """Computes the rotation matrix of the revolute joint.

    Args:
        r (Float[Array, "2 1"]): Generalized shape coordinates of the system

    Returns:
        Float[Array, "3 3"]: The rotation matrix of the revolute joint.
    """
    # Taking generalized coordinates r as the angles of the revolute joint
    # r = [theta_1, theta_2]
    return jnp.array(
        [
            [jnp.cos(r[1]), 0, jnp.sin(r[1])],
            [0, 1, 0],
            [-jnp.sin(r[1]), 0, jnp.cos(r[1])],
        ]
    ) @ jnp.array(
        [
            [1, 0, 0],
            [0, jnp.cos(r[0]), -jnp.sin(r[0])],
            [0, jnp.sin(r[0]), jnp.cos(r[0])],
        ]
    )


def rho_B(r: Float[Array, "2 1"], params: DynamicsParams) -> Float[Array, "3 1"]:
    """Computes the position vector of the COM of body B relative to the base body.

    Args:
        r (Float[Array, "2 1"]): Generalized shape coordinates of the system

    Returns:
        Float[Array, "3 1"]: The position vector of the COM of body B relative to the base body.
    """
    return params.rho_A_joint + R_joint(r) @ params.rho_joint_B


# Compute the partial derivative of rho_B with respect to r
partial_rho_B_partial_r = jax.jit(jax.grad(rho_B, argnums=0))


def J_B(r: Float[Array, "2 1"], params: DynamicsParams) -> Float[Array, "3 3"]:
    """Computes the inertia matrix of body B.

    Args:
        r (Float[Array, "2 1"]): Generalized shape coordinates of the system

    Returns:
        Float[Array, "3 3"]: The inertia matrix of body B.
    """
    # Taking generalized coordinates r as the angles of the revolute joint
    # r = [theta_1, theta_2]
    # We can compute the rotation matrix that the joint will apply to J_B
    # R_joint = rotx(theta_1) * roty(theta_2)
    R_joint_val = R_joint(r)
    return R_joint_val @ params.J_B @ R_joint_val.T


def skew(v: Float[Array, "3 1"]) -> Float[Array, "3 3"]:
    """Computes the skew-symmetric matrix of a vector.

    Args:
        v (Float[Array, "3 1"]): The input vector.

    Returns:
        Float[Array, "3 3"]: The skew-symmetric matrix.
    """
    return jnp.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def f(
    t: float, x: Float[Array, "17 1"], tau: Float[Array, "9 1"], params: DynamicsParams
) -> Float[Array, "17 1"]:
    # x = [x_A, mrp_A, r, v_A, omega_A, r_dot, m_A]
    x_A = x[:3]
    mrp_A = x[3:6]
    r = x[6:8]
    v_A = x[8:11]
    omega_A = x[11:14]
    r_dot = x[14:16]
    m_A = x[16]

    # tau = [tau_t, tau_a, tau_s]
    tau_t = tau[:3]
    tau_a = tau[3:6]
    tau_s = tau[6:]

    # Compute the rotation matrix
    R_A = mrp2rotm(mrp_A)

    # Compute the total mass matrix
    M = (m_A + params.m_B) * jnp.eye(3)

    # Compute rho_B
    rho_B_val = rho_B(r, params)
    rho_B_skew = skew(rho_B_val)

    partial_rho_B_partial_r_val = partial_rho_B_partial_r(r, params)

    J_B_val = J_B(r, params)
    J_tilde = params.J_A + params.m_B * rho_B_skew.T @ rho_B_skew + J_B_val
    m_tilde = (
        params.m_B * jnp.dot(partial_rho_B_partial_r_val, partial_rho_B_partial_r_val)
        + J_B_val
    )
    K = -params.m_B * rho_B_skew
    B_t = params.m_B * partial_rho_B_partial_r_val
    B_a = params.m_B * rho_B_skew * partial_rho_B_partial_r_val + J_B_val
    BB = jnp.block([[B_t], [B_a]])
    # TODO: Determine how to compute d/dt(BB). For now assume it is zero
    BB_dot = jnp.zeros_like(BB)

    JJ = jnp.block([[M, K], [K.T, J_tilde]])
    # TODO: Determine how to compute d/dt(JJ). For now assume it is zero
    JJ_dot = jnp.zeros_like(JJ)

    J_tilde_inv = jnp.linalg.inv(J_tilde)
    J_tt_inv = jnp.linalg.inv(M - K @ J_tilde_inv @ K.T)
    J_ta_inv = -J_tt_inv @ K @ J_tilde_inv
    M_inv = jnp.linalg.inv(M)
    J_aa_inv = jnp.linalg.inv(J_tilde - K.T @ M_inv @ K)
    J_at_inv = -M_inv @ J_aa_inv @ K.T

    JJ_inv = jnp.block([[J_tt_inv, J_ta_inv], [J_at_inv, J_aa_inv]])

    A_t = J_tt_inv @ (B_t - K @ J_tilde_inv @ B_a)
    A_a = J_aa_inv @ (B_a - K.T @ M_inv @ B_t)
    AA = jnp.block([[A_t], [A_a]])

    # Momenta (linear, angular)
    P = M @ v_A + K @ omega_A + B_t @ r_dot
    P_skew = skew(P)
    Omega = K.T @ v_A + J_tilde @ omega_A + B_a @ r_dot
    Omega_skew = skew(Omega)

    Q_ta = (
        JJ_inv
        @ (jnp.block([[jnp.zeros(3, 3), P_skew], [P_skew, Omega_skew]]) - JJ_dot)
        @ jnp.block([[v_A], [omega_A]])
        - JJ_inv @ BB_dot @ r_dot
        + jnp.block([[skew(omega_A) @ v_A], [jnp.zeros(3, 1)]])
    )

    # Q_t is the top block of Q, Q_a is the bottom block
    Q_t = Q_ta[:3]
    Q_a = Q_ta[3:]

    Q = (
        AA.T @ jnp.block([[jnp.zeros(3, 3), P_skew], [P_skew, Omega_skew]])
        - AA.T @ JJ_dot
        + BB_dot
    ) @ jnp.block([[v_A], [omega_A]])

    # Compute the generalized shape control input
    # TODO: Include partial V partial r, zeros for now
    u_s = tau_s - jnp.zeros(2, 1) - Q

    x_ddot = R_A @ (Q_t + A_t @ u_s)
    R_dot = R_A @ skew(omega_A)
    omega_dot = Q_a + A_a @ u_s


if __name__ == "__main__":
    pass
