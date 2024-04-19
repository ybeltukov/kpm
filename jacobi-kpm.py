from __future__ import annotations

from typing import Iterator

# note: an interactive matplotlib backend is expected
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import eval_jacobi, gamma, poch, roots_jacobi


def inv_h(α: float, β: float, K: int) -> np.ndarray:
    "Returns a (K,)-shaped array: 1/h_k, where h_k is the norm of the kth Jacobi polynomial."
    # a naive calculation, like
    # gamma(k+α+1)*gamma(k+β+1)/gamma(k+α+β+1)/math.factorial(k),
    # would fail since each gamma can exceed float limits.
    # so we use a ratio of two Pochhammer symbols instead.

    k = np.arange(K)
    multipliers = poch(k + β + 1, α) / poch(k + 1, α) * (2 * k + α + β + 1) / 2 ** (α + β + 1)
    # a special case is needed for k=0 and α+β+1=0
    if np.allclose(α + β + 1, 0):
        multipliers[0] = 1 / gamma(α + 1) / gamma(β + 1)
    return multipliers


def eval_many_jacobi_gen(x: np.ndarray, n: int, α: float, β: float) -> Iterator[np.ndarray]:
    "Yields n x-shaped arrays - the values of P_0 ... P_{n-1} at points x"
    assert n >= 1, n
    res_a, res_b = np.ones_like(x), (α - β) / 2 + (α + β + 2) / 2 * x
    yield res_a
    if n == 1:
        return
    yield res_b

    for k in range(2, n):
        a = k + α
        b = k + β
        c = a + b
        res_new = (
            (c - 1) * (c * (c - 2) * (x * res_b) + (a - b) * (c - 2 * k) * res_b)
            #
            - 2 * (a - 1) * (b - 1) * c * res_a
        ) / (2 * k * (c - k) * (c - 2))
        res_a, res_b = res_b, res_new
        yield res_b


def jacobi_g(α: float, β: float, N: int) -> np.ndarray:
    """
    Calculates g by quadrature from the expression for the optimal kernel.
    Returns an array of shape (N,): g_0, ..., g_{N-1}.
    """

    odd = N % 2 == 1
    M = (N + 1) // 2
    x_N, w_N = roots_jacobi(N, α, β)
    # largest root of P(α,β,M) or P(α,β+1,M):
    xi = (roots_jacobi(M, α, β) if odd else roots_jacobi(M, α, β + 1))[0][-1]

    # normalization constant:
    def C(α, β):
        return (1 - xi**2) / (2 * M + α + β + 1) * inv_h(α, β, M + 1)[M]

    def K(x):
        if odd:
            return C(α, β) * (eval_jacobi(M, α, β, x) / (x - xi)) ** 2
        else:
            return (1 + x) * C(α, β + 1) * (eval_jacobi(M, α, β + 1, x) / (x - xi)) ** 2

    kernel_vals = K(x_N)  # (N,)
    gs = np.zeros((N,), dtype=np.float64)
    P_gen = eval_many_jacobi_gen(np.concatenate([x_N, [1]]), N, α, β)  # generator of (N+1,) arrays
    for n, pvals in enumerate(P_gen):
        Pnk, P1 = pvals[:-1], pvals[-1]
        gs[n] = np.sum(kernel_vals * Pnk * w_N) / P1

    return gs


if __name__ == "__main__":
    # Generate some sparse dynamical matrix
    D = 2000
    m = 5000
    A = scipy.sparse.rand(D, m, 0.001, random_state=0, dtype=np.float64)
    A.data -= 0.5
    M = A @ A.T  # (n,n)

    reps = 30  # a number of repetitions to average the result
    omega_max = 1.9  # omega_max should be bigger than any frequency in the system
    deg = 200  # polynomial degree, which gives the frequency resolution ~omega_max/deg
    omega_pts = 1000  # number of points to evaluate the VDOS on. should be on the order of deg

    assert deg > 1

    # affine-transform M into a linear operator with eigenvalues laying within [-1,1]
    M_tilde = M * (2 / omega_max**2) - scipy.sparse.identity(D)

    # stochastically calculate the KPM moments
    m = np.zeros(deg, dtype=np.float64)
    α, β = (1 / 2, 1 / 2)
    for i in range(reps):
        v0 = np.random.randn(D)
        v0 /= np.linalg.norm(v0)
        u0 = v0
        u1 = (α - β) / 2 * v0 + (α + β + 2) / 2 * M_tilde @ v0
        m[0] += u0 @ v0
        m[1] += u1 @ v0
        for k in range(2, deg):
            a = k + α
            b = k + β
            c = a + b
            u_new = (
                (c - 1) * (c * (c - 2) * (M_tilde @ u1) + (a - b) * (c - 2 * k) * u1)
                #
                - 2 * (a - 1) * (b - 1) * c * u0
            ) / (2 * k * (c - k) * (c - 2))
            u1, u0 = u_new, u1
            m[k] += u1 @ v0
    m /= reps
    assert np.abs(m[0] - 1) < 1e-6, m[0]
    assert np.abs(m).max() < 1e3, np.abs(m).max()

    # apply damping factors g and divide by norms h
    m *= jacobi_g(α, β, deg) * inv_h(α, β, deg)

    # Calculate the ω and ε values to evaluate the DOS at.
    # We space the points uniformly in the frequency space, which is uneven in ε-space
    omega = np.linspace(0, omega_max, omega_pts)
    x = omega**2
    eps = x * (2 / omega_max**2) - 1
    # Calculate the DOS:
    rho = np.zeros_like(eps)
    p1 = (α - β) / 2 * np.ones_like(eps) + (α + β + 2) / 2 * eps  # P_1(x)
    res1, res0 = p1, np.ones_like(eps)
    rho += res0 * m[0]
    rho += res1 * m[1]
    for k in range(2, deg):
        a = k + α
        b = k + β
        c = a + b
        res_new = (
            (c - 1) * (c * (c - 2) * (eps * res1) + (a - b) * (c - 2 * k) * res1)
            #
            - 2 * (a - 1) * (b - 1) * c * res0
        ) / (2 * k * (c - k) * (c - 2))
        res1, res0 = res_new, res1
        rho += res1 * m[k]

    rho *= (1 - eps) ** α * (1 + eps) ** β

    # Go from DOS of M_tilde ρ(ε) to DOS of M ρ(x)
    rho /= omega_max**2 / 2

    # Convert DOS into VDOS
    g = 2 * omega * rho

    # Plot the resulting VDOS
    plt.figure()
    plt.plot(omega, g)
    plt.xlabel("Frequency")
    plt.ylabel("VDOS")
    # Compare with the histogram of eigenfrequencies
    plt.hist(np.sqrt(np.abs(np.linalg.eigvalsh(M.toarray()))), 50, density=True)
    plt.show()
