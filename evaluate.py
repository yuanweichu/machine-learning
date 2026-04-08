"""
论文配图生成脚本 v2.0
数据配置：10,000 Samples, 3-Run Average
CVPR 学术风格 (包含误差棒)
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

# ============================================
# 最新实验数据 (10,000 Samples)
# ============================================
models = ['SVM + HOG', 'CNN']

# 准确率平均值与标准差
acc_mean = [55.13, 58.40]
acc_std = [0.54, 0.99]

# 时间成本平均值 (秒)
time_mean = [17.37, 15.79]

# 内存消耗 (取第一次实验的峰值或代表性均值)
memory_val = [71.57, 72.97] 

x = np.arange(len(models))
width = 0.45
colors = ['#2E86AB', '#A23B72'] # 高对比度蓝/紫

# ============================================
# 图1：准确率对比图 (带误差棒)
# ============================================
fig1, ax1 = plt.subplots(figsize=(6, 5))

bars1 = ax1.bar(x, acc_mean, width, yerr=acc_std, 
                color=colors, edgecolor='black', linewidth=1.2,
                capsize=10, error_kw={'elinewidth':2, 'ecolor':'#333333'})

ax1.set_ylabel('Accuracy (%)', fontweight='bold')
ax1.set_title('Accuracy Comparison (n=10000)', fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylim(0, 70) # 调高上限以适应更高的准确率
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random Guess (50%)')

for bar, acc in zip(bars1, acc_mean):
    height = bar.get_height()
    ax1.annotate(f'{acc:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height + 2), # 避开误差棒
                textcoords="offset points", xytext=(0, 5),
                ha='center', va='bottom', fontweight='bold')

ax1.legend(loc='upper right')
plt.tight_layout()
fig1.savefig('accuracy_comparison_v2.png')
print("已保存: accuracy_comparison_v2.png")

# ============================================
# 图2：训练时间对比图
# ============================================
fig2, ax2 = plt.subplots(figsize=(6, 5))

bars2 = ax2.bar(x, time_mean, width, color=colors, edgecolor='black', linewidth=1.2)

ax2.set_ylabel('Total Time (Seconds)', fontweight='bold')
ax2.set_title('Computational Efficiency (n=10000)', fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylim(0, 25)

for bar, t in zip(bars2, time_mean):
    height = bar.get_height()
    ax2.annotate(f'{t:.2f}s',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
fig2.savefig('time_comparison_v2.png')
print("已保存: time_comparison_v2.png")

# ============================================
# 图3：内存消耗对比图
# ============================================
fig3, ax3 = plt.subplots(figsize=(6, 5))

bars3 = ax3.bar(x, memory_val, width, color=colors, edgecolor='black', linewidth=1.2)

ax3.set_ylabel('Peak Memory (MB)', fontweight='bold')
ax3.set_title('Memory Consumption Comparison', fontweight='bold', pad=15)
ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.set_ylim(0, 100)

for bar, mem in zip(bars3, memory_val):
    height = bar.get_height()
    ax3.annotate(f'{mem:.2f} MB',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
fig3.savefig('memory_comparison_v2.png')
print("已保存: memory_comparison_v2.png")

plt.close('all')