"""
phase_portraits.py

Спрощений виконуваний скрипт для лабораторної роботи:
- знаходження точок рівноваги
- лінеаризація (Якобіан), власні значення
- класифікація: вузол/сідло/фокус/центр/вироджена
- фазовий портрет: векторне поле, ізокліни f=0,g=0, інтегральні криві

Автор: ChatGPT (українська версія)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# ------------------ Визначення систем ------------------
# Кожна система: f(x,y), g(x,y), jac(x,y), (опційно) відомі рівноваги, область
SYSTEMS = {
    "sys1: x'=-y, y'=x-3x^2": {
        "f": lambda x, y: -y,
        "g": lambda x, y: x - 3 * x**2,
        "jac": lambda x, y: np.array([[0.0, -1.0], [1.0 - 6.0*x, 0.0]]),
        "known_eq": [(0.0, 0.0), (1.0/3.0, 0.0)],
        "x_range": (-1.5, 2.0), "y_range": (-2.0, 2.0),
        "description": "(0,0) — очікувано центр; (1/3,0) — сідло"
    },
    "sys2: x'=2y, y'=4x-4x^3": {
        "f": lambda x, y: 2*y,
        "g": lambda x, y: 4*x - 4*x**3,
        "jac": lambda x, y: np.array([[0.0, 2.0], [4.0 - 12.0*x**2, 0.0]]),
        "known_eq": [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0)],
        "x_range": (-2.0, 2.0), "y_range": (-2.0, 2.0),
        "description": "(0,0) — сідло; (±1,0) — центри"
    },
    "sys3: x'=-2xy, y'=x^2+y^2-1": {
        "f": lambda x, y: -2*x*y,
        "g": lambda x, y: x**2 + y**2 - 1.0,
        "jac": lambda x, y: np.array([[-2.0*y, -2.0*x], [2.0*x, 2.0*y]]),
        "known_eq": [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)],
        "x_range": (-2.0, 2.0), "y_range": (-2.0, 2.0),
        "description": "очікування: попарно седла/центри за лінеаризацією"
    }
}


# ------------------ Утиліти ------------------
def classify_equilibrium(J, tol=1e-8):
    """
    Класифікація за власними значеннями матриці J:
    - сідло: дійсні λ з різними знаками
    - вузол (стійкий/нестійкий): два дійсні λ однакового знака
    - фокус: компл. λ з невід'ємною/від'ємною дійсною частиною
    - центр: чисто уявні λ (реальні частини ~ 0)
    - вироджена: випадки з нульовими λ або кратними реальними із проблемами лінеаризації
    """
    vals = np.linalg.eigvals(J)
    re = np.real(vals)
    im = np.imag(vals)

    # якщо уявні частини близькі до нуля -> дійсні власні значення
    if np.all(np.abs(im) < tol):
        a, b = re[0], re[1]
        if a * b < 0:
            return "Сідло"
        if a > 0 and b > 0:
            return "Вузол (нестійкий)"
        if a < 0 and b < 0:
            return "Вузол (стійкий)"
        # випадок з одним нулем або дуже малим
        if np.isclose(a, 0, atol=tol) or np.isclose(b, 0, atol=tol):
            return "Вироджена (є нульовий λ)"
        return "Вироджена / невизначена"
    else:
        # комплексні власні значення
        if np.all(np.abs(re) < tol):
            return "Центр (маргінальний)"
        if np.mean(re) > 0:
            return "Фокус (нестійкий)"
        else:
            return "Фокус (стійкий)"


def find_equilibria_grid(f, g, x_rng, y_rng, nx=25, ny=25, tol=1e-8):
    """
    Простий автопошук рівноваг (початкові вгадки з сетки + fsolve).
    Повертає список унікальних рівноваг (ті, що збігаються до tol, вважаються однаковими).
    """
    xs = np.linspace(x_rng[0], x_rng[1], nx)
    ys = np.linspace(y_rng[0], y_rng[1], ny)
    found = []
    for xi in xs:
        for yi in ys:
            try:
                root = fsolve(lambda z: [f(z[0], z[1]), g(z[0], z[1])], x0=[xi, yi], xtol=1e-10, maxfev=200)
                rx, ry = float(root[0]), float(root[1])
                # перевірка (малий залишок)
                if abs(f(rx, ry)) + abs(g(rx, ry)) > 1e-6:
                    continue
                # унікальність
                if not any(np.hypot(rx - ex, ry - ey) < 1e-5 for ex, ey in found):
                    found.append((rx, ry))
            except Exception:
                pass
    return found


# ------------------ Візуалізація ------------------
def plot_phase_portrait(syst, nx_field=24, stream_seeds=(8, 8), t_span=8.0):
    f = syst["f"]
    g = syst["g"]
    jac = syst["jac"]
    x_min, x_max = syst["x_range"]
    y_min, y_max = syst["y_range"]

    # знаходимо рівноваги: спочатку автопошук, потім додаємо відомі (якщо є)
    auto_eqs = find_equilibria_grid(f, g, (x_min, x_max), (y_min, y_max), nx=18, ny=18)
    eqs = list(auto_eqs)
    for known in syst.get("known_eq", []):
        if not any(np.hypot(known[0]-ex, known[1]-ey) < 1e-6 for ex, ey in eqs):
            eqs.append((float(known[0]), float(known[1])))

    # Показ інформації у консолі
    print("Система:", syst.get("description", ""))
    print("Знайдені точки рівноваги:")
    for (ex, ey) in eqs:
        J = jac(ex, ey)
        vals = np.linalg.eigvals(J)
        typ = classify_equilibrium(J)
        print(f"  ({ex:.6g}, {ey:.6g})  — λ = {vals[0]:.6g}, {vals[1]:.6g}  → {typ}")

    # Сітка для векторного поля
    xs = np.linspace(x_min, x_max, nx_field)
    ys = np.linspace(y_min, y_max, nx_field)
    X, Y = np.meshgrid(xs, ys)
    U = f(X, Y)
    V = g(X, Y)

    # нормалізація стрілок для однакового вигляду
    speed = np.sqrt(U**2 + V**2)
    U_n = U / (speed + 1e-9)
    V_n = V / (speed + 1e-9)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(syst.get("description", "Фазовий портрет"))
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # векторне поле
    ax.quiver(X, Y, U_n, V_n, angles='xy', scale_units='xy', scale=15, width=0.003, alpha=0.7)

    # ізокліни f=0 та g=0 (контур)
    # для чіткості беремо більш щільну сітку
    xs_f = np.linspace(x_min, x_max, 200)
    ys_f = np.linspace(y_min, y_max, 200)
    XX, YY = np.meshgrid(xs_f, ys_f)
    CF = ax.contour(XX, YY, f(XX, YY), levels=[0], colors='red', linewidths=1.5)
    CG = ax.contour(XX, YY, g(XX, YY), levels=[0], colors='blue', linewidths=1.5)
    ax.clabel(CF, fmt={0: "f=0"}, inline=True, fontsize=8)
    ax.clabel(CG, fmt={0: "g=0"}, inline=True, fontsize=8)

    # інтегральні криві (streamlines) — інтегруємо прямування вперед і назад
    def rhs(t, z):
        return [f(z[0], z[1]), g(z[0], z[1])]

    # насіння із регулярної сітки (нижче пораховано); уникаємо потрапляння прямо в точки рівноваги
    nx_s, ny_s = stream_seeds
    seeds_x = np.linspace(x_min * 0.9 + 0.1 * x_max, x_max * 0.9 + 0.1 * x_min, nx_s)
    seeds_y = np.linspace(y_min * 0.9 + 0.1 * y_max, y_max * 0.9 + 0.1 * y_min, ny_s)
    seeds = [(sx, sy) for sx in seeds_x for sy in seeds_y]

    # видалимо насіння занадто близько до рівноваг
    filtered_seeds = []
    for sx, sy in seeds:
        if any(np.hypot(sx-ex, sy-ey) < 1e-2 for ex, ey in eqs):
            continue
        filtered_seeds.append((sx, sy))

    for (sx, sy) in filtered_seeds:
        # вперед
        sol_f = solve_ivp(rhs, (0.0, t_span), (sx, sy), max_step=0.05, rtol=1e-6, atol=1e-9)
        # назад
        sol_b = solve_ivp(rhs, (0.0, -t_span), (sx, sy), max_step=0.05, rtol=1e-6, atol=1e-9)
        xs_traj = np.hstack([sol_b.y[0][::-1], sol_f.y[0]])
        ys_traj = np.hstack([sol_b.y[1][::-1], sol_f.y[1]])
        ax.plot(xs_traj, ys_traj, lw=0.6, color='k', alpha=0.55)

    # позначаємо точки рівноваги
    for (ex, ey) in eqs:
        J = jac(ex, ey)
        vals = np.linalg.eigvals(J)
        typ = classify_equilibrium(J)
        ax.plot(ex, ey, 'o', color='magenta', markersize=8)
        ax.text(ex + 0.03 * (x_max-x_min), ey + 0.03 * (y_max-y_min),
                f"{typ}\nλ={vals[0]:.3g}, {vals[1]:.3g}", fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    plt.show()


# ------------------ Головний запуск ------------------
if __name__ == "__main__":
    print("Доступні системи:")
    keys = list(SYSTEMS.keys())
    for i, k in enumerate(keys, 1):
        print(f" {i}. {k} — {SYSTEMS[k].get('description','')}")
    print()

    # Вибір системи: ввід від користувача (індекс або назва)
    sel = input(f"Оберіть систему (1-{len(keys)}, або назва) [за замовчуванням 1]: ").strip()
    if sel == "":
        idx = 1
    else:
        try:
            idx = int(sel)
        except ValueError:
            # знайдемо за назвою
            matches = [i+1 for i, k in enumerate(keys) if sel.lower() in k.lower()]
            idx = matches[0] if matches else 1

    chosen_key = keys[idx-1]
    print(f"\nОбрано: {chosen_key}\n")

    plot_phase_portrait(SYSTEMS[chosen_key])
