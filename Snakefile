import numpy as np
import h5py as h5
from numba import njit


@njit
def grad2(x: np.array):
    m, n = x.shape
    y = np.zeros_like(x)
    for i in range(m):
        for j in range(n):
            y[i,j] = x[(i-1)%m,j] + x[i,(j-1)%n] \
                   + x[(i+1)%m,j] + x[i,(j+1)%n] - 4*x[i,j]
    return y


def euler_method(df, y_init, t_init, t_end, t_step):
    n_steps = int((t_end - t_init) / t_step)
    y = y_init
    times = (t_init + i*t_step for i in range(n_steps))
    return (y := y + df(y, t)*t_step for t in times)


def initial_state(shape) -> np.ndarray:
    U = np.ones(shape, dtype=np.float32)
    V = np.zeros(shape, dtype=np.float32)

    centre = (slice(shape[0]//2-10, shape[0]//2+10),
              slice(shape[1]//2-10, shape[1]//2+10))
    U[centre] = 1/2
    V[centre] = 1/4

    U += np.random.normal(0.0, 0.01, size=shape)
    V += np.random.normal(0.0, 0.01, size=shape)

    return np.stack((U, V))


def reaction_diffusion(F, k, D_u=2e-5, D_v=1e-5, res=0.01):
    def df(state: np.ndarray, _: float) -> np.ndarray:
        U, V = state
        du = D_u*grad2(U)/res**2 - U*V**2 + F*(1 - U)
        dv = D_v*grad2(V)/res**2 + U*V**2 - (F + k)*V
        return np.stack((du, dv))
    return df


def run_model(k, F, t_end=10_000, write_interval=20, shape=(256, 256)):
    n_snaps = t_end // write_interval
    result = np.zeros(shape=[n_snaps, 2, shape[0], shape[1]],
                      dtype=np.float32)

    rd = reaction_diffusion(k=k, F=F)
    init = initial_state(shape=shape)
    comp = euler_method(rd, init, 0, t_end, 1)
    for i, snap in enumerate(comp):
        if i % write_interval == 0:
            result[i // write_interval] = snap
    
    return result


k_values = np.linspace(0.03, 0.07, 11)
F_values = np.linspace(0.00, 0.08, 11)


rule map_vis:
    input:
        expand("k{param_k}-F{param_F}.h5",
               param_k=[f"{k:02}" for k in range(len(k_values))],
               param_F=[f"{F:02}" for F in range(len(F_values))])
    output:
        "pattern_map.png"
    run:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(len(k_values), len(F_values), figsize=(20, 20))
        for fname in input:
            with h5.File(fname, "r") as f_in:
                i = f_in.attrs["k_index"]
                j = f_in.attrs["F_index"]
                ax[i,j].imshow(f_in["V"][-1])
                ax[i,j].set_axis_off()
        # fig.tight_layout()
        fig.savefig(output[0], bbox_inches="tight")


rule compute_model:
    output:
        "k{param_k}-F{param_F}.h5"
    run:
        k = k_values[int(wildcards.param_k)]
        F = F_values[int(wildcards.param_F)]
        result = run_model(k, F)
        with h5.File(f"{output[0]}", "w") as f_out:
            f_out.attrs["k"] = k
            f_out.attrs["F"] = F
            f_out.attrs["k_index"] = int(wildcards.param_k)
            f_out.attrs["F_index"] = int(wildcards.param_F)
            f_out["U"] = result[:, 0]
            f_out["V"] = result[:, 1]

