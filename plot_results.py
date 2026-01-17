#!/usr/bin/env python3
"""
Script pour analyser et visualiser les résultats de la multiplication matrix-matrix
Génère 4 graphiques:
1. Temps d'exécution (ms)
2. GFLOPS (performance)
3. Speedup comparé au séquentiel
4. Comparaison OpenMP vs Hybride
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Vérifier que le fichier CSV existe
if not Path('metrics.csv').exists():
    print("ERROR: metrics.csv not found!")
    print("Please run './run_all.sh' first")
    exit(1)

# Charger les données
df = pd.read_csv('metrics.csv')

print("=" * 60)
print("RESULTS ANALYSIS")
print("=" * 60)
print("\nData loaded:")
print(df.to_string())
print()

# Statistiques de base
print("\nSUMMARY STATISTICS:")
print("-" * 60)
for version in df['version'].unique():
    data = df[df['version'] == version]
    avg_time = data['time_ms'].mean()
    avg_gflops = data['gflops'].mean()
    print(f"{version:20s} | Time: {avg_time:8.2f} ms | GFLOPS: {avg_gflops:8.2f}")

# Créer les graphiques
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Matrix-Matrix Multiplication Performance Analysis\n(2000×2000)', 
             fontsize=16, fontweight='bold')

# --- Graphique 1: Temps d'exécution ---
ax1 = axes[0, 0]
colors = []
labels = []
times = []
for version in df['version'].unique():
    data = df[df['version'] == version]
    avg_time = data['time_ms'].mean()
    times.append(avg_time)
    labels.append(version)
    
    if 'Sequential' in version:
        colors.append('red')
    elif 'OpenMP' in version and 'Hybrid' not in version:
        colors.append('blue')
    elif 'MPI' in version and 'Hybrid' not in version:
        colors.append('green')
    else:
        colors.append('orange')

x = np.arange(len(labels))
bars1 = ax1.bar(x, times, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Version', fontweight='bold')
ax1.set_ylabel('Time (ms)', fontweight='bold')
ax1.set_title('Execution Time Comparison', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3)

# Ajouter les valeurs sur les barres
for i, (bar, time) in enumerate(zip(bars1, times)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{time:.1f}', ha='center', va='bottom', fontsize=9)

# --- Graphique 2: GFLOPS ---
ax2 = axes[0, 1]
gflops_list = []
for version in df['version'].unique():
    data = df[df['version'] == version]
    avg_gflops = data['gflops'].mean()
    gflops_list.append(avg_gflops)

bars2 = ax2.bar(x, gflops_list, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Version', fontweight='bold')
ax2.set_ylabel('GFLOPS', fontweight='bold')
ax2.set_title('Performance (GFLOPS)', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

# Ajouter les valeurs sur les barres
for i, (bar, gflops) in enumerate(zip(bars2, gflops_list)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{gflops:.1f}', ha='center', va='bottom', fontsize=9)

# --- Graphique 3: Speedup vs Séquentiel ---
ax3 = axes[1, 0]
seq_time = df[df['version'] == 'Sequential']['time_ms'].values[0]
speedups = []
speedup_labels = []
speedup_colors = []

for version in df['version'].unique():
    if version != 'Sequential':
        data = df[df['version'] == version]
        avg_time = data['time_ms'].mean()
        speedup = seq_time / avg_time
        speedups.append(speedup)
        speedup_labels.append(version)
        
        if 'OpenMP' in version and 'Hybrid' not in version:
            speedup_colors.append('blue')
        elif 'MPI' in version and 'Hybrid' not in version:
            speedup_colors.append('green')
        else:
            speedup_colors.append('orange')

x_speedup = np.arange(len(speedup_labels))
bars3 = ax3.bar(x_speedup, speedups, color=speedup_colors, alpha=0.7, edgecolor='black')
ax3.set_xlabel('Version', fontweight='bold')
ax3.set_ylabel('Speedup', fontweight='bold')
ax3.set_title('Speedup vs Sequential', fontweight='bold')
ax3.set_xticks(x_speedup)
ax3.set_xticklabels(speedup_labels, rotation=45, ha='right')
ax3.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline (1x)')
ax3.grid(axis='y', alpha=0.3)
ax3.legend()

# Ajouter les valeurs sur les barres
for bar, speedup in zip(bars3, speedups):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{speedup:.2f}x', ha='center', va='bottom', fontsize=9)

# --- Graphique 4: OpenMP scaling (si disponible) ---
ax4 = axes[1, 1]
omp_data = df[df['version'].str.contains('OpenMP_', na=False)].copy()

if not omp_data.empty:
    omp_data['threads'] = omp_data['num_threads'].astype(int)
    omp_data = omp_data.sort_values('threads')
    
    threads = omp_data['threads'].values
    times_omp = omp_data['time_ms'].values
    gflops_omp = omp_data['gflops'].values
    
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(threads, times_omp, 'o-', color='blue', linewidth=2, 
                     markersize=8, label='Execution Time')
    line2 = ax4_twin.plot(threads, gflops_omp, 's-', color='green', linewidth=2, 
                          markersize=8, label='GFLOPS')
    
    ax4.set_xlabel('Number of Threads', fontweight='bold')
    ax4.set_ylabel('Time (ms)', color='blue', fontweight='bold')
    ax4_twin.set_ylabel('GFLOPS', color='green', fontweight='bold')
    ax4.set_title('OpenMP Scaling', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='green')
    
    lines = line1 + line2
    labels_legend = [l.get_label() for l in lines]
    ax4.legend(lines, labels_legend, loc='upper left')
else:
    ax4.text(0.5, 0.5, 'No OpenMP scaling data available', 
             ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('OpenMP Scaling', fontweight='bold')

plt.tight_layout()
plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Graph saved to 'performance_analysis.png'")
plt.show()

# Générer un rapport texte
print("\n" + "=" * 60)
print("DETAILED REPORT")
print("=" * 60)

with open('performance_report.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("MATRIX-MATRIX MULTIPLICATION PERFORMANCE REPORT\n")
    f.write("Matrix size: 2000 × 2000\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("SUMMARY STATISTICS:\n")
    f.write("-" * 60 + "\n")
    for version in df['version'].unique():
        data = df[df['version'] == version]
        avg_time = data['time_ms'].mean()
        avg_gflops = data['gflops'].mean()
        avg_throughput = data['throughput_gb_s'].mean()
        f.write(f"{version:25s} | Time: {avg_time:8.2f} ms | GFLOPS: {avg_gflops:8.2f} | Throughput: {avg_throughput:6.2f} GB/s\n")
    
    f.write("\n" + "=" * 60 + "\n")
    f.write("SPEEDUP ANALYSIS (vs Sequential)\n")
    f.write("-" * 60 + "\n")
    f.write(f"Sequential baseline: {seq_time:.2f} ms\n\n")
    
    for version, speedup in zip(speedup_labels, speedups):
        f.write(f"{version:25s} | Speedup: {speedup:.2f}x\n")
    
    f.write("\n" + "=" * 60 + "\n")
    f.write("ANALYSIS & INSIGHTS\n")
    f.write("-" * 60 + "\n")
    
    # Analyser les résultats
    best_version = df.loc[df['time_ms'].idxmin(), 'version']
    best_gflops = df.loc[df['gflops'].idxmax(), 'version']
    best_speedup_idx = speedups.index(max(speedups)) if speedups else -1
    
    f.write(f"\n1. Best execution time: {best_version}\n")
    f.write(f"2. Best GFLOPS performance: {best_gflops}\n")
    if best_speedup_idx >= 0:
        f.write(f"3. Best speedup: {speedup_labels[best_speedup_idx]} ({max(speedups):.2f}x)\n")
    
    # Analyse OpenMP
    if not omp_data.empty:
        f.write("\n4. OpenMP Scaling Analysis:\n")
        min_time = omp_data['time_ms'].min()
        max_time = omp_data['time_ms'].max()
        best_thread_count = omp_data.loc[omp_data['time_ms'].idxmin(), 'threads']
        f.write(f"   - Best performance with {int(best_thread_count)} threads\n")
        f.write(f"   - Improvement: {max_time - min_time:.2f} ms ({100*(max_time-min_time)/max_time:.1f}%)\n")
    
    f.write("\n" + "=" * 60 + "\n")
    f.write("RECOMMENDATIONS\n")
    f.write("-" * 60 + "\n")
    f.write("1. For maximum performance, use the hybrid approach\n")
    f.write("2. Balance MPI ranks and OpenMP threads based on system resources\n")
    f.write("3. Monitor NUMA effects on multi-socket systems\n")
    f.write("4. Use proper work scheduling (dynamic, guided) for better load balancing\n")

print("✓ Detailed report saved to 'performance_report.txt'")
print("\nGenerated files:")
print("  - performance_analysis.png")
print("  - performance_report.txt")
print("  - metrics.csv")