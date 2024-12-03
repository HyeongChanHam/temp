import numpy as np
import plotly.graph_objects as go
from scipy.stats import invwishart

# Inverse Wishart 분포 매개변수 설정
df = 5  # 자유도
scale_matrix = np.array([[2, 0.5], [0.5, 1]])  # 스케일 행렬

# Inverse Wishart에서 샘플 생성
num_samples = 10  # 표시할 타원 수
samples = invwishart.rvs(df, scale_matrix, size=num_samples if num_samples > 1 else None)

# 각 샘플의 고유값, 고유벡터, 확률 계산
ellipses = []
probabilities = []

for sample in (samples if num_samples > 1 else [samples]):
    # 고유값과 고유벡터 계산
    eigenvalues, eigenvectors = np.linalg.eigh(sample)
    long_axis = np.sqrt(eigenvalues[1])  # 장축 길이
    short_axis = np.sqrt(eigenvalues[0])  # 단축 길이
    angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])  # 회전 각도 (라디안)

    # 확률 밀도 계산
    prob = invwishart.pdf(sample, df, scale_matrix)
    probabilities.append(prob)

    # 타원 정보 저장
    ellipses.append((long_axis, short_axis, angle, prob))

# Plotly 시각화 데이터 생성
fig = go.Figure()

# 타원을 추가
for (long_axis, short_axis, angle, prob) in ellipses:
    theta = np.linspace(0, 2 * np.pi, 100)
    x = long_axis * np.cos(theta)
    y = short_axis * np.sin(theta)

    # 회전 적용
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle),  np.cos(angle)]])
    points = np.dot(rotation_matrix, np.array([x, y]))
    x_rot, y_rot = points

    # 타원 추가 (xy 평면, z = prob)
    fig.add_trace(go.Scatter3d(
        x=x_rot, y=y_rot, z=[prob] * len(x_rot),
        mode='lines',
        line=dict(color='blue', width=2),
        name=f"Probability: {prob:.3f}"
    ))

# 원점에서 z축으로 향하는 직선 추가
max_z = max(probabilities)
fig.add_trace(go.Scatter3d(
    x=[0, 0], y=[0, 0], z=[0, max_z],
    mode='lines',
    line=dict(color='black', width=5),
    name="z-axis Line"
))

# 그래프 레이아웃 설정
fig.update_layout(
    title="Interactive 3D Visualization of Inverse Wishart Samples",
    scene=dict(
        xaxis_title="X-axis (Long Axis)",
        yaxis_title="Y-axis (Short Axis)",
        zaxis_title="Probability Density",
    ),
    showlegend=True
)

# 그래프 출력
fig.show()
