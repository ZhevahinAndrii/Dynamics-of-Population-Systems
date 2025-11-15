import numpy as np
import matplotlib.pyplot as plt

def linear_phase_portrait(a, b, c, d,
                          x_range=(-4, 4),
                          y_range=(-4, 4),
                          grid_size=25,
                          t_max=10,
                          h=0.02,
                          title="Лінійна система"):

    A = np.array([[a, b],
                  [c, d]], dtype=float)

    def f(z):
        return A @ z

    def solve_trajectory(z0):
        t = 0
        z = np.array(z0, dtype=float)
        path = [z.copy()]
        while t < t_max:
            z = z + h * f(z)
            path.append(z.copy())
            t += h
        return np.array(path)

    
    xs = np.linspace(x_range[0], x_range[1], grid_size)
    ys = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(xs, ys)

    U = a * X + b * Y
    V = c * X + d * Y

    plt.figure(figsize=(6,6))
    plt.quiver(X, Y, U, V, color='tab:blue', alpha=0.7)

    initials = [(2,0), (-2,1), (1.5,-1.5), (-1.5,-2)]
    for p in initials:
        path = solve_trajectory(p)
        plt.plot(path[:,0], path[:,1], lw=1.8)

    eigvals = np.linalg.eigvals(A)

    plt.title(title + f"\nВласні значення: {np.round(eigvals,3)}")
    plt.axhline(0, color='black', lw=0.6)
    plt.axvline(0, color='black', lw=0.6)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, ls='--', alpha=0.5)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.tight_layout()


linear_phase_portrait(a=3, b=2,
                      c=-4, d=-1,
                      title="Нестійкий фокус: x' = 3x+2y, y' = -4x-y")


plt.show()
