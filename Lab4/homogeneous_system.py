import numpy as np
import matplotlib.pyplot as plt


class HomogeneousSystem:

    def __init__(self, degree, P_coeffs, Q_coeffs):
        """
        degree — степінь однорідності m\n 
        P_coeffs[k] — коефіцієнт при x^(m-k) y^k у P(x,y)\n
        Q_coeffs[k] — коефіцієнт при x^(m-k) y^k у Q(x,y)
        """
        
        assert len(P_coeffs) == len(Q_coeffs), "Функції не одного степеня"
        self.m = len(P_coeffs) - 1
        self.Pc = np.array(P_coeffs, dtype=float)
        self.Qc = np.array(Q_coeffs, dtype=float)
        
    def P(self, x, y):
        """Поліном P(x,y)."""
        res = np.zeros_like(x, dtype=float)
        for k, c in enumerate(self.Pc):
            res += c * (x ** (self.m - k)) * (y ** k)
        return res

    def Q(self, x, y):
        """Поліном Q(x,y)."""
        res = np.zeros_like(x, dtype=float)
        for k, c in enumerate(self.Qc):
            res += c * (x ** (self.m - k)) * (y ** k)
        return res

    def AB(self, phi):
        """A(φ) та B(φ) для полярної діагностики."""
        cosφ = np.cos(phi)
        sinφ = np.sin(phi)

        A = np.zeros_like(phi)
        B = np.zeros_like(phi)

        for k in range(self.m + 1):
            A += self.Qc[k] * (cosφ ** (self.m - k)) * (sinφ ** k)
            B += self.Pc[k] * (cosφ ** (self.m - k)) * (sinφ ** k)

        return A, B

    def plot_phase(self, lim:float=3, density:int=2, diagnose:bool|None=True, title=None):

        grid = np.linspace(-lim, lim, 400)
        X, Y = np.meshgrid(grid, grid)

        U = self.P(X, Y)
        V = self.Q(X, Y)

        plt.figure(figsize=(7, 7))
        plt.streamplot(X, Y, U, V, density=density)
        plt.axhline(0, color='k', lw=0.6)
        plt.axvline(0, color='k', lw=0.6)

        plt.title(title or f"Однорідна система степеня m={self.m}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.tight_layout()

        if diagnose:
            phi = np.linspace(0, 2*np.pi, 1500)
            A, B = self.AB(phi)

            N = A * np.cos(phi) - B * np.sin(phi)
            Z = A * np.sin(phi) + B * np.cos(phi)

            eps = 1e-10
            R = np.where(abs(N) < eps, np.nan, Z / N)

            plt.figure(figsize=(7, 5))
            plt.plot(phi, R, lw=1.6)

            zero_idx = np.where(np.diff(np.signbit(N)))[0]
            plt.scatter(phi[zero_idx], np.zeros_like(zero_idx),
                        s=18, color="red", label="N(φ)=0")

            plt.grid(True)
            plt.title("Полярна діагностика R(φ)")
            plt.xlabel("φ")
            plt.ylabel("R(φ)")
            plt.legend()
            plt.tight_layout()


# P(x, y) = x^3 - 3xy^2
P_coeffs = [1, 0, -3, 0]

# Q(x, y) = 3x^2y - y^3
Q_coeffs = [0, 3, 0, -1]

sys3 = HomogeneousSystem(3, P_coeffs, Q_coeffs)

sys3.plot_phase(
    lim = 4,
    title="Кубічна однорідна система"
)

plt.show()
