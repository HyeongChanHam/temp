import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from matplotlib.widgets import Slider

def update_plot(mean_x, var_x, mean_y, var_y, point_x, point_y, alpha):
    # 설정한 2차원 정규분포의 평균과 공분산 행렬 정의
    mean = [mean_x, mean_y]
    cov = [[var_x, 0], [0, var_y]]
    rv = multivariate_normal(mean, cov)
    
    # 다변량 정규분포 샘플링
    samples = rv.rvs(size=100000)
    
    # 임의의 점과 각 샘플 사이의 거리 계산
    point = np.array([point_x, point_y])
    distances = np.linalg.norm(samples - point, axis=1)
    
    # 거리 기반 확률분포 계산
    transformed_distances = 10 - distances
    hist, bin_edges = np.histogram(transformed_distances, bins=100, range=(0, 10), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # CVaR 계산
    sorted_transformed_distances = np.sort(transformed_distances)
    var_index = int((1 - alpha) * len(sorted_transformed_distances))
    var = sorted_transformed_distances[var_index]
    cvar = np.mean(sorted_transformed_distances[var_index:])
    
    # 그래프 갱신
    ax[0].clear()
    ax[1].clear()
    
    # 첫 번째 서브플롯: 다변량 정규분포
    ax[0].scatter(point_x, point_y, c='red', label='Point', zorder=5)
    ax[0].scatter(mean_x, mean_y, c='blue', marker='x', label='Mean', zorder=5)
    
    # 1-sigma 타원 추가
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigenvalues)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  edgecolor='blue', facecolor='none', linestyle='--', linewidth=2, zorder=5)
    ax[0].add_patch(ell)
    
    ax[0].set_xlim(0, 10)
    ax[0].set_ylim(0, 10)
    ax[0].set_xlabel("X-axis")
    ax[0].set_ylabel("Y-axis")
    ax[0].set_title("2D Multivariate Normal Distribution")
    ax[0].legend()
    
    # 두 번째 서브플롯: 거리 기반 확률분포와 CVaR
    ax[1].plot(bin_centers, hist, label="P(10 - d)", color='blue')
    ax[1].axvline(var, color='green', linestyle='--', label=f"VaR (α={alpha}) = {var:.2f}")
    ax[1].axvline(cvar, color='red', linestyle='-', label=f"CVaR = {cvar:.2f}")
    ax[1].set_xlabel("10 - d")
    ax[1].set_ylabel("Probability Density")
    ax[1].set_title("Transformed Distance Distribution (10 - d)")
    ax[1].legend()
    
    fig.canvas.draw_idle()

# 초기 값
initial_mean_x = 4
initial_var_x = 2
initial_mean_y = 6
initial_var_y = 3
initial_point_x = 5
initial_point_y = 5
initial_alpha = 0.95

# 그래프 설정
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.3)

# 슬라이더 설정
ax_mean_x = plt.axes([0.2, 0.2, 0.6, 0.03])
ax_var_x = plt.axes([0.2, 0.15, 0.6, 0.03])
ax_mean_y = plt.axes([0.2, 0.1, 0.6, 0.03])
ax_var_y = plt.axes([0.2, 0.05, 0.6, 0.03])
ax_point_x = plt.axes([0.2, 0.25, 0.6, 0.03])
ax_point_y = plt.axes([0.2, 0.3, 0.6, 0.03])
ax_alpha = plt.axes([0.2, 0.35, 0.6, 0.03])

slider_mean_x = Slider(ax_mean_x, 'Mean X', 3, 5, valinit=initial_mean_x)
slider_var_x = Slider(ax_var_x, 'Variance X', 1, 3, valinit=initial_var_x)
slider_mean_y = Slider(ax_mean_y, 'Mean Y', 5, 8, valinit=initial_mean_y)
slider_var_y = Slider(ax_var_y, 'Variance Y', 2, 4, valinit=initial_var_y)
slider_point_x = Slider(ax_point_x, 'Point X', 0, 10, valinit=initial_point_x)
slider_point_y = Slider(ax_point_y, 'Point Y', 0, 10, valinit=initial_point_y)
slider_alpha = Slider(ax_alpha, 'Alpha', 0.8, 0.99, valinit=initial_alpha)

# 슬라이더 업데이트
def update(val):
    update_plot(
        slider_mean_x.val,
        slider_var_x.val,
        slider_mean_y.val,
        slider_var_y.val,
        slider_point_x.val,
        slider_point_y.val,
        slider_alpha.val
    )

slider_mean_x.on_changed(update)
slider_var_x.on_changed(update)
slider_mean_y.on_changed(update)
slider_var_y.on_changed(update)
slider_point_x.on_changed(update)
slider_point_y.on_changed(update)
slider_alpha.on_changed(update)

# 초기 플롯
update_plot(
    initial_mean_x,
    initial_var_x,
    initial_mean_y,
    initial_var_y,
    initial_point_x,
    initial_point_y,
    initial_alpha
)

plt.show()
