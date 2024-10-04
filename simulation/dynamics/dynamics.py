from functools import partial

import jax
import jax.numpy as jnp
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve
from dynamicsparams import params, x_0
from jaxtyping import Array, Float


@jax.jit
def mrp2rotm(mrp: Float[Array, "3 1"]) -> Float[Array, "3 3"]:
    """Converts an MRP vector (3x1) to a rotation matrix (3x3).

    Taken from https://github.com/mlourakis/MRPs/blob/master/mrp2rot.m

    Returns:
        Float[Array, "3 3"]: The rotation matrix.
    """
    sqmag = mrp.T @ mrp
    skw = skew(mrp)
    R = (
        jnp.eye(3) + ((4 * (1 - sqmag) * jnp.eye(3) + 8 * skw) @ skw) / (1 + sqmag) ** 2
    )  # (255b) from Shuster

    return R


@jax.jit
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

    return mrp.reshape(-1, 1)


@jax.jit
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
            [jnp.cos(r[1, 0]), 0, jnp.sin(r[1, 0])],
            [0, 1, 0],
            [-jnp.sin(r[1, 0]), 0, jnp.cos(r[1, 0])],
        ]
    ) @ jnp.array(
        [
            [1, 0, 0],
            [0, jnp.cos(r[0, 0]), -jnp.sin(r[0, 0])],
            [0, jnp.sin(r[0, 0]), jnp.cos(r[0, 0])],
        ]
    )


@jax.jit
def rho_B(r: Float[Array, "2 1"]) -> Float[Array, "3 1"]:
    """Computes the position vector of the COM of body B relative to the base body.

    Args:
        r (Float[Array, "2 1"]): Generalized shape coordinates of the system

    Returns:
        Float[Array, "3 1"]: The position vector of the COM of body B relative to the base body.
    """
    return params.rho_A_joint + R_joint(r) @ params.rho_joint_B


# Compute the partial derivative of rho_B with respect to r
partial_rho_B_partial_r = jax.jit(jax.jacfwd(rho_B, argnums=0))


@jax.jit
def J_B(r: Float[Array, "2 1"]) -> Float[Array, "3 3"]:
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


@jax.jit
def skew(v: Float[Array, "3 1"]) -> Float[Array, "3 3"]:
    """Computes the skew-symmetric matrix of a vector.

    Args:
        v (Float[Array, "3 1"]): The input vector.

    Returns:
        Float[Array, "3 3"]: The skew-symmetric matrix.
    """
    return jnp.array(
        [
            [0, -v[2, 0], v[1, 0]],
            [v[2, 0], 0, -v[0, 0]],
            [-v[1, 0], v[0, 0], 0],
        ]
    )


@jax.jit
def BB(r: Float[Array, "2 1"]) -> Float[Array, "6 2"]:
    # Compute rho_B
    rho_B_val = rho_B(r)
    rho_B_skew = skew(rho_B_val.reshape(-1, 1))

    partial_rho_B_partial_r_val = jnp.squeeze(partial_rho_B_partial_r(r))

    J_B_val = J_B(r)
    B_t = params.m_B * partial_rho_B_partial_r_val
    B_a = (params.m_B * rho_B_skew @ partial_rho_B_partial_r_val) + J_B_val @ params.C_B
    BB = jnp.block([[B_t], [B_a]])
    return BB


partial_BB_partial_r = jax.jit(jax.jacfwd(BB, argnums=0))


@jax.jit
def JJ(r: Float[Array, "2 1"], m_A: Float[Array, "1 1"]) -> Float[Array, "6 6"]:
    # Compute the total mass matrix
    M = (m_A + params.m_B) * jnp.eye(3)

    # Compute rho_B
    rho_B_val = rho_B(r)
    rho_B_skew = skew(rho_B_val.reshape(-1, 1))

    J_B_val = J_B(r)
    J_tilde = params.J_A + params.m_B * rho_B_skew.T @ rho_B_skew + J_B_val
    K = -params.m_B * rho_B_skew

    JJ = jnp.block([[M, K], [K.T, J_tilde]])
    return JJ


partial_JJ_partial_r = jax.jit(jax.jacfwd(JJ, argnums=0))
partial_JJ_partial_m_A = jax.jit(jax.jacfwd(JJ, argnums=1))


@jax.jit
def m_tilde(r: Float[Array, "2 1"]) -> Float[Array, "3 3"]:
    partial_rho_B_partial_r_val = jnp.squeeze(partial_rho_B_partial_r(r))

    J_B_val = J_B(r)
    m_tilde = (
        params.m_B * partial_rho_B_partial_r_val.T @ partial_rho_B_partial_r_val
        + params.C_B.T @ J_B_val @ params.C_B
    )
    return m_tilde


partial_m_tilde_partial_r = jax.jit(jax.jacfwd(m_tilde, argnums=0))


@jax.jit
def T(
    r: Float[Array, "2 1"],
    r_dot: Float[Array, "2 1"],
    v_A: Float[Array, "3 1"],
    omega_A: Float[Array, "3 1"],
    m_A: Float[Array, "1 1"],
) -> Float[Array, "1"]:
    # Compute the total mass matrix
    M = (m_A + params.m_B) * jnp.eye(3)

    # Compute rho_B
    rho_B_val = rho_B(r)
    rho_B_skew = skew(rho_B_val.reshape(-1, 1))

    partial_rho_B_partial_r_val = jnp.squeeze(partial_rho_B_partial_r(r))

    J_B_val = J_B(r)
    J_tilde = params.J_A + params.m_B * rho_B_skew.T @ rho_B_skew + J_B_val
    m_tilde = (
        params.m_B * partial_rho_B_partial_r_val.T @ partial_rho_B_partial_r_val
        + params.C_B.T @ J_B_val @ params.C_B
    )
    K = -params.m_B * rho_B_skew
    B_t = params.m_B * partial_rho_B_partial_r_val
    B_a = (params.m_B * rho_B_skew @ partial_rho_B_partial_r_val) + J_B_val @ params.C_B
    return jnp.squeeze(
        0.5
        * (
            jnp.block([[v_A], [omega_A], [r_dot]]).T
            @ jnp.block([[M, K, B_t], [K.T, J_tilde, B_a], [B_t.T, B_a.T, m_tilde]])
            @ jnp.block([[v_A], [omega_A], [r_dot]])
        )
    )


partial_T_partial_r = jax.jit(jax.grad(T, argnums=0))


@jax.jit
def POmega2vomega(
    P: Float[Array, "3 1"],
    Omega: Float[Array, "3 1"],
    r_dot: Float[Array, "2 1"],
    J_tt_inv: Float[Array, "3 3"],
    J_ta_inv: Float[Array, "3 3"],
    J_at_inv: Float[Array, "3 3"],
    J_aa_inv: Float[Array, "3 3"],
    A_t: Float[Array, "3 3"],
    A_a: Float[Array, "3 3"],
) -> Float[Array, "6 1"]:
    """Converts the linear and angular momenta to linear and angular velocities.

    The linear system is given by:
    P = M @ v_A + K @ omega_A + B_t @ r_dot
    Omega = K.T @ v_A + J_tilde @ omega_A + B_a @ r_dot

    Or:
    [P]     = [M   K      ] [v_A]       + [B_t] [r_dot]
    [Omega]   [K.T J_tilde] [omega_A]     [B_a]

    Returns:
        Float[Array, "6 1"]: The linear and angular velocities concatenated.
    """
    v = J_tt_inv @ P + J_ta_inv @ Omega - A_t @ r_dot
    omega = J_at_inv @ P + J_aa_inv @ Omega - A_a @ r_dot

    return jnp.concatenate([v, omega])


@jax.jit
def V(r: Float[Array, "2 1"]) -> Float[Array, "1"]:
    """Potential energy of the system.

    Potential energy is given by mg(delta_h) where delta_h is the height of the COM of body B
    relative to where it would be at the equilibrium position (r = [0, 0])

    Returns:
        Float[Array, "1"]: The potential energy.
    """
    rho_B_val = rho_B(r)
    rho_B_base_val = rho_B(jnp.array([[0.0], [0.0]]))
    delta_h = rho_B_val[2, 0] - rho_B_base_val[2, 0]
    return params.m_B * 9.81 * delta_h


partial_V_partial_r = jax.jit(jax.grad(V, argnums=0))


@jax.jit
def f(
    t: float, x: Float[Array, "18 1"], tau: Float[Array, "9 1"]
) -> Float[Array, "18 1"]:
    # x = [x_A, mrp_A, r, v_A, omega_A, r_dot, m_A]
    mrp_A = x[3:6, :]
    r = x[6:8, :]
    P_A = x[8:11, :]
    Omega_A = x[11:14, :]
    r_dot = x[14:16, :]
    m_A = x[16, :]
    m_A_dot = x[17, :]

    # tau = [tau_t, tau_a, tau_s]
    tau_t = tau[:3, :]
    tau_a = tau[3:6, :]
    tau_s = tau[6:, :]

    # Compute the rotation matrix
    R_A = mrp2rotm(mrp_A)

    # Compute the total mass matrix
    M = (m_A + params.m_B) * jnp.eye(3)

    # Compute rho_B
    rho_B_val = rho_B(r)
    rho_B_skew = skew(rho_B_val.reshape(-1, 1))

    partial_rho_B_partial_r_val = jnp.squeeze(partial_rho_B_partial_r(r))

    J_B_val = J_B(r)
    J_tilde = params.J_A + params.m_B * rho_B_skew.T @ rho_B_skew + J_B_val
    m_tilde = (
        params.m_B * partial_rho_B_partial_r_val.T @ partial_rho_B_partial_r_val
        + params.C_B.T @ J_B_val @ params.C_B
    )
    m_tilde_dot = jnp.squeeze(jnp.squeeze(partial_m_tilde_partial_r(r)) @ r_dot)
    K = -params.m_B * rho_B_skew
    B_t = params.m_B * partial_rho_B_partial_r_val
    B_a = (params.m_B * rho_B_skew @ partial_rho_B_partial_r_val) + J_B_val @ params.C_B
    BB_dot = jnp.squeeze(jnp.squeeze(partial_BB_partial_r(r)) @ r_dot)

    JJ_dot = jnp.squeeze(
        jnp.squeeze(partial_JJ_partial_r(r, m_A)) @ r_dot
    ) + jnp.squeeze(jnp.squeeze(partial_JJ_partial_m_A(r, m_A)) * m_A_dot)

    J_tilde_inv = jnp.linalg.inv(J_tilde)
    J_tt_inv = jnp.linalg.inv(M - K @ J_tilde_inv @ K.T)
    J_ta_inv = -J_tt_inv @ K @ J_tilde_inv
    M_inv = jnp.linalg.inv(M)
    J_aa_inv = jnp.linalg.inv(J_tilde - K.T @ M_inv @ K)
    J_at_inv = -M_inv @ J_aa_inv @ K.T

    # JJ_inv = jnp.block([[J_tt_inv, J_ta_inv], [J_at_inv, J_aa_inv]])

    A_t = J_tt_inv @ (B_t - K @ J_tilde_inv @ B_a)
    A_a = J_aa_inv @ (B_a - K.T @ M_inv @ B_t)
    AA = jnp.block([[A_t], [A_a]])

    # Compute the velocity and angular velocity of body A
    vomega = POmega2vomega(
        P_A, Omega_A, r_dot, J_tt_inv, J_ta_inv, J_at_inv, J_aa_inv, A_t, A_a
    )
    v_A = vomega[:3, :]
    omega_A = vomega[3:, :]

    # Momenta (linear, angular)
    P = M @ v_A + K @ omega_A + B_t @ r_dot
    P_skew = skew(P)
    Omega = K.T @ v_A + J_tilde @ omega_A + B_a @ r_dot
    Omega_skew = skew(Omega)

    Q = (
        (
            AA.T @ jnp.block([[jnp.zeros((3, 3)), P_skew], [P_skew, Omega_skew]])
            - AA.T @ JJ_dot
            + BB_dot.T
        )
        @ jnp.block([[v_A], [omega_A]])
        + (m_tilde_dot - AA.T @ BB_dot) @ r_dot
        - partial_T_partial_r(r, r_dot, v_A, omega_A, m_A)
    )

    P_dot = P_skew @ omega_A + tau_t
    Omega_dot = P_skew @ v_A + Omega_skew @ omega_A + tau_a

    mrp_A_dot = (
        0.5
        * (0.5 * (1 - mrp_A.T @ mrp_A) * jnp.eye(3) + skew(mrp_A) + mrp_A @ mrp_A.T)
        @ omega_A
    )  # https://ntrs.nasa.gov/api/citations/19960035754/downloads/19960035754.pdf

    x_A_dot = R_A @ v_A

    # partial_V_partial_r_val = jnp.squeeze(partial_V_partial_r(r))
    partial_V_partial_r_val = jnp.array([[0.0], [0.0]])
    r_ddot = (
        jnp.linalg.inv(m_tilde - AA.T @ JJ(r, m_A) @ AA)
        @ jnp.squeeze(
            tau_s - A_t.T @ tau_t - A_a.T @ tau_a - partial_V_partial_r_val - Q
        )
    ).reshape(-1, 1)

    return jnp.concatenate(
        [
            x_A_dot,
            mrp_A_dot,
            r_dot,
            P_dot,
            Omega_dot,
            r_ddot,
            m_A_dot.reshape(-1, 1),
            jnp.array([[0]]),
        ]
    )


if __name__ == "__main__":
    tau = jnp.array([[0.0, 0.0, 10.0, 0.0, 0.05, 0.0, 0.0, 0.0]]).T
    f_val = f(0.0, x_0, tau)

    # Extract the values
    x_A = f_val[:3, :]
    mrp_A = f_val[3:6, :]
    r = f_val[6:8, :]
    P_A = f_val[8:11, :]
    Omega_A = f_val[11:14, :]
    r_dot = f_val[14:16, :]
    m_A = f_val[16, :]
    m_A_dot = f_val[17, :]

    print(f"x_A: {x_A}")
    print(f"mrp_A: {mrp_A}")
    print(f"r: {r}")
    print(f"P_A: {P_A}")
    print(f"Omega_A: {Omega_A}")
    print(f"r_dot: {r_dot}")
    print(f"m_A: {m_A}")
    print(f"m_A_dot: {m_A_dot}")

    # Solve differential equation
    term = ODETerm(f)
    solver = Tsit5()
    import numpy as np

    ts = np.linspace(0, 35, 100)
    saveat = SaveAt(ts=ts)
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

    import time

    start_time = time.time()
    sol = diffeqsolve(
        term,
        solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=0.05,
        y0=x_0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        args=tau,
        max_steps=None,
    )
    print(f"Elapsed time: {time.time() - start_time}")
    print(sol.ts)
    print(sol.ys)

    last_mrp_A = sol.ys[-1, 3:6, 0]
    # Convert to rotation matrix
    R_A = mrp2rotm(last_mrp_A.reshape(-1, 1))

    import matplotlib.pyplot as plt

    # Plot x, z
    plt.figure()
    plt.plot(sol.ts, sol.ys[:, 0, 0], label="x")
    plt.plot(sol.ts, sol.ys[:, 2, 0], label="z")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend()
    plt.show()
