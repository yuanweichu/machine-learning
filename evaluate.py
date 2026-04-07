"""
论文配图生成脚本
SVM vs CNN 对比实验可视化
CVPR 学术风格
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置学术风格
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# 实验数据
models = ['SVM + HOG', 'CNN']
accuracy = [51.40, 53.40]
time_cost = [2.46, 8.45]
memory = [35.29, 60.80]

x = np.arange(len(models))
width = 0.5

# 颜色配置 - 适合黑白打印的高对比度方案
colors = ['#2E86AB', '#A23B72']


# ============================================
# 图1：准确率对比图
# ============================================
fig1, ax1 = plt.subplots(figsize=(6, 5))

bars1 = ax1.bar(x, accuracy, width, color=colors, edgecolor='black', linewidth=1.2)

ax1.set_ylabel('Accuracy (%)', fontweight='bold')
ax1.set_title('Accuracy Comparison: SVM vs CNN', fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylim(0, 60)
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

for bar, acc in zip(bars1, accuracy):
    height = bar.get_height()
    ax1.annotate(f'{acc:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontweight='bold', fontsize=12)

ax1.legend(['Baseline (50%)'], loc='upper right', framealpha=0.9)
plt.tight_layout()
fig1.savefig('accuracy_comparison.png', bbox_inches='tight', facecolor='white')
print("已保存: accuracy_comparison.png")
plt.close(fig1)


# ============================================
# 图2：训练时间对比图
# ============================================
fig2, ax2 = plt.subplots(figsize=(6, 5))

bars2 = ax2.bar(x, time_cost, width, color=colors, edgecolor='black', linewidth=1.2)

ax2.set_ylabel('Time (Seconds)', fontweight='bold')
ax2.set_title('Training Time Comparison: SVM vs CNN', fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylim(0, 10)

for bar, t in zip(bars2, time_cost):
    height = bar.get_height()
    ax2.annotate(f'{t:.2f}s',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontweight='bold', fontsize=12)

ax2.legend(['Slower'], loc='upper left', framealpha=0.9)
plt.tight_layout()
fig2.savefig('time_comparison.png', bbox_inches='tight', facecolor='white')
print("已保存: time_comparison.png")
plt.close(fig2)


# ============================================
# 图3：内存消耗对比图
# ============================================
fig3, ax3 = plt.subplots(figsize=(6, 5))

bars3 = ax3.bar(x, memory, width, color=colors, edgecolor='black', linewidth=1.2)

ax3.set_ylabel('Memory (MB)', fontweight='bold')
ax3.set_title('Memory Consumption Comparison: SVM vs CNN', fontweight='bold', pad=15)
ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.set_ylim(0, 70)

for bar, mem in zip(bars3, memory):
    height = bar.get_height()
    ax3.annotate(f'{mem:.2f} MB',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontweight='bold', fontsize=12)

ax3.legend(['Higher Memory'], loc='upper left', framealpha=0.9)
plt.tight_layout()
fig3.savefig('memory_comparison.png', bbox_inches='tight', facecolor='white')
print("已保存: memory_comparison.png")
plt.close(fig3)


# ============================================
# 总结
# ============================================
print("\n" + "=" * 50)
print("所有论文配图已生成完成！")
print("=" * 50)
print("\n生成的图表文件:")
print("  1. accuracy_comparison.png  - 准确率对比")
print("  2. time_comparison.png      - 训练时间对比")
print("  3. memory_comparison.png   - 内存消耗对比")
print("\n图表特点:")
print("  - 适合黑白打印的高对比度配色")
print("  - 包含清晰的坐标轴标签和数值标注")
print("  - 采用 CVPR 学术风格")
print("=" * 50)