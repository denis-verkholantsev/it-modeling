
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def equation1(u):
    return u

def equation2(y, u):
    return -(p / m) * u - (k / m) * y

def runge_kutta(h, y, u):
    k1_y = h * equation1(u)
    k1_u = h * equation2(y, u)
    
    k2_y = h * equation1(u + 0.5 * k1_u)
    k2_u = h * equation2(y + 0.5 * k1_y, u + 0.5 * k1_u)

    k3_y = h *  equation1(u + 0.5 * k2_u)
    k3_u = h * equation2(y + 0.5 * k2_y, u + 0.5 * k2_u)

    k4_y = h * equation1(u + k3_u)
    k4_u = h * equation2(y + k3_y, u + k3_u)

    return y + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6, u + (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6

# решение методом Рунге-Кутты
def RK(y_0, u_0, t_range, h):
    ys = np.zeros(len(t_range))
    us = np.zeros(len(t_range))
    ys[0], us[0] = y_0, u_0

    for i in range(len(t_range) - 1):
        ys[i + 1], us[i + 1] = runge_kutta(h, ys[i], us[i])

    return ys, us


def system(_, vals):
    y, u = vals
    return [u, -p/m * u - k/m * y]


# решение методом Рунге-Кутты из пакета scipy
def ivpsolve(y_0, u_0, t_span, t_range):
    solution = solve_ivp(fun=system, t_span=t_span, y0=[y_0, u_0], t_eval=t_range, dense_output=True)
    return solution.y[0], solution.y[1]


def plot(ivp_ys, ivp_us, ys, us, t_range):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    ax1.plot(t_range, ivp_ys, label='y(t)')
    ax1.plot(t_range, ivp_us, label='dy/dt')
    ax1.set_title('my solution')
    ax1.set_xlabel('t')
    ax1.set_ylabel('f(t)')
    ax1.legend()
    ax1.grid()

    ax2.plot(t_range, ys, label='y(t)')
    ax2.plot(t_range, us, label="dy/dt")
    ax2.set_title('scipy')
    ax2.set_xlabel('t')
    ax2.set_ylabel('f(t)')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()


# сравнение графиков
def plot_compare(ys, ivp_ys, label, label_ivp):
    plt.figure(figsize=(10,14))
    plt.title(f'Compare {label} and {label_ivp}')
    plt.plot(t_range, ivp_ys, label=label_ivp, color='red', marker='o', markersize=5)
    plt.plot(t_range, ys, label=label, color='blue', marker='.', markersize=2)
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.grid()
    plt.legend()
    plt.show()


p, m, k = 1, 2, 4
y_0, u_0 = 1, 2
h = 0.001
t_span = (-10, 10)
t_range = np.arange(start=t_span[0], stop=t_span[1], step=h)

ys, us = RK(y_0, u_0, t_range, h)
ivp_ys, ivp_us = ivpsolve(y_0, u_0, t_span, t_range)

plot(ivp_ys, ivp_us, ys, us, t_range)
plot_compare(ys, ivp_ys, 'y(t)', 'scipy y(t)')
plot_compare(us, ivp_us, 'dy/dt', 'scipy dy/dt')


