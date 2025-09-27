import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress

# --------------------- Вхідні дані ---------------------
years = 50

initial_population = np.array([900, 630, 745, 910])
birth_rates = np.array([0.4, 2.6, 3.8, 0.4])
survival_rates = np.array([0.4, 0.8, 0.65, 0.0])

# --------------------- Матриця Леслі ---------------------
leslie_matrix = np.array([
    birth_rates,
    [survival_rates[0], 0, 0, 0],
    [0, survival_rates[1], 0, 0],
    [0, 0, survival_rates[2], 0]
])

# --------------------- Моделювання ---------------------
populations = [initial_population]
for t in range(1, years+1):
    next_population = leslie_matrix @ populations[-1]
    populations.append(next_population)

populations = np.array(populations)
print(populations[-1])
total_population = populations.sum(axis=1)
log_population = np.log(total_population + 1)  # +1 щоб уникнути log(0)
age_labels = ["0-1 рік", "1-2 роки", "2-3 роки", "3-<4 роки"]


# --------------------- Візуалізація ---------------------
fig = make_subplots(rows=3, cols=1, 
                    subplot_titles=("Динаміка по вікових групах", 
                                    "Загальна чисельність", 
                                    "Логарифм чисельності (тренд)"))


for i in range(4):
    fig.add_trace(go.Scatter(y=populations[:, i], mode="lines", name=age_labels[i]), row=1, col=1)

fig.add_trace(go.Scatter(y=total_population, mode="lines+markers", name="Всього"), row=2, col=1)

# Логарифм чисельності
fig.add_trace(go.Scatter(y=log_population, mode="lines", name="log(N)"), row=3, col=1)

fig.update_layout(
    height=900,
    title="Модель динаміки популяції мишей (матриця Леслі)",
    xaxis_title="Роки",
    template="plotly_white"
)

fig.show()

# --------------------- Аналіз стійкості ---------------------
eigenvalues, _ = np.linalg.eig(leslie_matrix)
lambda_max = max(np.real(eigenvalues))

# Тренд через регресію log(N)
slope, intercept, r_value, p_value, std_err = linregress(range(len(log_population)), log_population)

print("========== АНАЛІЗ СТІЙКОСТІ ==========")
if lambda_max > 1:
    print(f"Популяція в довгостроковій перспективі зростає (λ = {lambda_max:.3f})")
elif lambda_max < 1:
    print(f"Популяція в довгостроковій перспективі зменшується (λ = {lambda_max:.3f})")
else:
    print(f"Популяція стабільна (λ = {lambda_max:.3f})")
print("========== АНАЛІЗ ТРЕНДУ НА ОЗНАКУ ЗРОСТАННЯ ==========")

if slope > 0:
    print(f"Тренд загальної чисельності → зростання (нахил = {slope:.3f})")
elif slope < 0:
    print(f"Тренд загальної чисельності → спадання (нахил = {slope:.3f})")
else:
    print("Тренд загальної чисельності стабільний")
