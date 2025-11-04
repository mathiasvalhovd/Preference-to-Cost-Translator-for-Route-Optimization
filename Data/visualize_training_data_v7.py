"""
Visualization Function for V7 Training Data (with sample weights)

Updates from original:
1. Handles sample_weight field
2. Shows weight distribution analysis
3. Shows preference diversity analysis
4. Shows cost diversity analysis
5. Color-codes by sample weight (importance)
"""

def visualize_training_data_v7(training_data_file, cost_pool_file):
    """
    Create comprehensive visualizations for V7 training data
    
    Creates folder: {training_data_file_stem}/
    Saves: 
      - pareto_visualization.png (3D + 2D projections)
      - usage_analysis.png (solution usage, Pareto vs non-Pareto)
      - weight_analysis.png (NEW: sample weight distribution)
      - preference_analysis.png (NEW: preference space coverage)
      - cost_diversity.png (NEW: cost parameter distributions)
      - statistics.txt (comprehensive stats)
    """
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from collections import Counter
    from pathlib import Path
    
    # Use the same folder as training data
    output_dir = Path(training_data_file).parent
    print(f"\n📁 Creating visualizations in: {output_dir}/")
    
    # Load data
    with open(cost_pool_file, 'r') as f:
        cost_pool = json.load(f)
    
    with open(training_data_file, 'r') as f:
        training_data = json.load(f)
    
    print(f"  Cost pool: {len(cost_pool)} solutions")
    print(f"  Training data: {len(training_data)} samples")
    
    # Check if V7 format
    has_weights = 'sample_weight' in training_data[0]
    print(f"  Format: {'V7 (with weights)' if has_weights else 'V6 (no weights)'}")
    
    # Extract features
    all_parking = [s['features']['avg_parking_difficulty'] for s in cost_pool]
    all_time = [s['features']['total_travel_hours'] for s in cost_pool]
    all_distance = [s['features']['total_distance_km'] for s in cost_pool]
    
    # Get pool_ids used in training
    training_pool_ids = set([s['metadata']['pool_id'] for s in training_data])
    
    # Separate into used/unused
    used_parking = [s['features']['avg_parking_difficulty'] for s in cost_pool if s['pool_id'] in training_pool_ids]
    used_time = [s['features']['total_travel_hours'] for s in cost_pool if s['pool_id'] in training_pool_ids]
    used_distance = [s['features']['total_distance_km'] for s in cost_pool if s['pool_id'] in training_pool_ids]
    
    unused_parking = [s['features']['avg_parking_difficulty'] for s in cost_pool if s['pool_id'] not in training_pool_ids]
    unused_time = [s['features']['total_travel_hours'] for s in cost_pool if s['pool_id'] not in training_pool_ids]
    unused_distance = [s['features']['total_distance_km'] for s in cost_pool if s['pool_id'] not in training_pool_ids]
    
    # Find Pareto front
    class SimpleParetoFinder:
        def dominates(self, a, b):
            better_in_one = False
            for key in ['avg_parking_difficulty', 'total_travel_hours', 'total_distance_km']:
                if a['features'][key] > b['features'][key]:
                    return False
                if a['features'][key] < b['features'][key]:
                    better_in_one = True
            return better_in_one
        
        def find_pareto_front(self, solutions):
            pareto = []
            for candidate in solutions:
                if not any(self.dominates(other, candidate) 
                          for other in solutions 
                          if other['pool_id'] != candidate['pool_id']):
                    pareto.append(candidate)
            return pareto
    
    finder = SimpleParetoFinder()
    pareto_front = finder.find_pareto_front(cost_pool)
    
    pareto_parking = [s['features']['avg_parking_difficulty'] for s in pareto_front]
    pareto_time = [s['features']['total_travel_hours'] for s in pareto_front]
    pareto_distance = [s['features']['total_distance_km'] for s in pareto_front]
    pareto_ids = [s['pool_id'] for s in pareto_front]
    
    print(f"  Pareto front: {len(pareto_front)} solutions")
    print(f"  Used in training: {len(training_pool_ids)} unique solutions")
    
    # =========================================================================
    # VISUALIZATION 1: 3D Scatter Plot + 2D Projections
    # =========================================================================
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: 3D view
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Unused solutions (gray)
    ax1.scatter(unused_parking, unused_time, unused_distance, 
               c='lightgray', s=30, alpha=0.3, label='Unused in training')
    
    # Used solutions (blue)
    ax1.scatter(used_parking, used_time, used_distance,
               c='steelblue', s=100, alpha=0.7, edgecolors='black', 
               linewidths=1, label=f'Used in training ({len(training_pool_ids)})')
    
    # Pareto front (red stars)
    ax1.scatter(pareto_parking, pareto_time, pareto_distance,
               c='red', s=300, marker='*', edgecolors='black',
               linewidths=2, label=f'Pareto front ({len(pareto_front)})', zorder=10)
    
    ax1.set_xlabel('Parking Difficulty', fontsize=10)
    ax1.set_ylabel('Travel Time (hours)', fontsize=10)
    ax1.set_zlabel('Distance (km)', fontsize=10)
    ax1.set_title('3D Feature Space: Training Data + Pareto Front', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    
    # 2D Projections
    projections = [
        (2, 2, 2, 'avg_parking_difficulty', 'total_travel_hours', 'Parking vs Time'),
        (2, 2, 3, 'avg_parking_difficulty', 'total_distance_km', 'Parking vs Distance'),
        (2, 2, 4, 'total_travel_hours', 'total_distance_km', 'Time vs Distance')
    ]
    
    for row, col, pos, x_feat, y_feat, title in projections:
        ax = fig.add_subplot(row, col, pos) 
        
        # Extract data for this projection
        unused_x = [s['features'][x_feat] for s in cost_pool if s['pool_id'] not in training_pool_ids]
        unused_y = [s['features'][y_feat] for s in cost_pool if s['pool_id'] not in training_pool_ids]
        
        used_x = [s['features'][x_feat] for s in cost_pool if s['pool_id'] in training_pool_ids]
        used_y = [s['features'][y_feat] for s in cost_pool if s['pool_id'] in training_pool_ids]
        
        pareto_x = [s['features'][x_feat] for s in pareto_front]
        pareto_y = [s['features'][y_feat] for s in pareto_front]
        
        # Plot
        ax.scatter(unused_x, unused_y, c='lightgray', s=30, alpha=0.3, label='Unused')
        ax.scatter(used_x, used_y, c='steelblue', s=100, alpha=0.7, 
                  edgecolors='black', linewidths=1, label='Used')
        ax.scatter(pareto_x, pareto_y, c='red', s=300, marker='*',
                  edgecolors='black', linewidths=2, label='Pareto', zorder=10)
        
        ax.set_xlabel(x_feat.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel(y_feat.replace('_', ' ').title(), fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz1_path = Path(output_dir) / 'pareto_visualization.png'
    plt.savefig(viz1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {viz1_path}")
    
    # =========================================================================
    # VISUALIZATION 2: Usage Analysis
    # =========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Count how many times each solution is used
    pool_id_counts = Counter([s['metadata']['pool_id'] for s in training_data])
    
    # Left plot: Bar chart of usage
    ax = axes[0]
    pool_ids_sorted = sorted(pool_id_counts.keys())
    counts = [pool_id_counts[pid] for pid in pool_ids_sorted]
    colors = ['red' if pid in pareto_ids else 'steelblue' for pid in pool_ids_sorted]
    
    bars = ax.bar(range(len(pool_ids_sorted)), counts, color=colors, 
                 edgecolor='black', alpha=0.7)
    ax.set_xlabel('Pool ID', fontsize=12)
    ax.set_ylabel('Times Used in Training', fontsize=12)
    ax.set_title(f'Solution Usage Distribution\n(Red = Pareto front)', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(pool_ids_sorted)))
    ax.set_xticklabels([f'{pid}' for pid in pool_ids_sorted], rotation=90, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right plot: Pareto vs Non-Pareto usage
    ax = axes[1]
    
    pareto_usage = sum(pool_id_counts.get(pid, 0) for pid in pareto_ids)
    non_pareto_usage = len(training_data) - pareto_usage
    
    pareto_in_training = len([pid for pid in training_pool_ids if pid in pareto_ids])
    non_pareto_in_training = len(training_pool_ids) - pareto_in_training
    
    labels = [f'Pareto Front\n({pareto_in_training} solutions)', 
              f'Non-Pareto\n({non_pareto_in_training} solutions)']
    sizes = [pareto_usage, non_pareto_usage]
    colors_pie = ['red', 'steelblue']
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Training Data: Pareto vs Non-Pareto Usage', 
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    viz2_path = Path(output_dir) / 'usage_analysis.png'
    plt.savefig(viz2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {viz2_path}")
    
    # =========================================================================
    # NEW VISUALIZATION 3: Sample Weight Analysis (V7 only)
    # =========================================================================
    
    if has_weights:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        weights = np.array([s['sample_weight'] for s in training_data])
        distances = np.array([s['metadata']['pareto_distance'] for s in training_data])
        
        # Plot 1: Weight distribution histogram
        ax = axes[0, 0]
        ax.hist(weights, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.median(weights), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(weights):.2f}')
        ax.set_xlabel('Sample Weight', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Sample Weight Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Distance vs Weight scatter
        ax = axes[0, 1]
        ax.scatter(distances, weights, alpha=0.5, s=20, c='steelblue')
        ax.set_xlabel('Pareto Distance', fontsize=12)
        ax.set_ylabel('Sample Weight', fontsize=12)
        ax.set_title('Distance vs Weight (should be inversely related)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Rank vs Weight boxplot
        ax = axes[1, 0]
        ranks = [s['metadata']['rank_for_preference'] for s in training_data]
        unique_ranks = sorted(set(ranks))
        rank_weights = [[s['sample_weight'] for s in training_data if s['metadata']['rank_for_preference'] == r] 
                       for r in unique_ranks]
        
        bp = ax.boxplot(rank_weights, positions=unique_ranks, widths=0.6)
        ax.set_xlabel('Rank (1=best match)', fontsize=12)
        ax.set_ylabel('Sample Weight', fontsize=12)
        ax.set_title('Sample Weight by Rank', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(range(1, min(31, max(unique_ranks)+1), 5))
        
        # Plot 4: Cumulative weight by rank
        ax = axes[1, 1]
        cumsum_weights = []
        for r in unique_ranks:
            total_weight = sum(s['sample_weight'] for s in training_data if s['metadata']['rank_for_preference'] == r)
            cumsum_weights.append(total_weight)
        
        cumsum_weights = np.cumsum(cumsum_weights)
        cumsum_pct = cumsum_weights / cumsum_weights[-1] * 100
        
        ax.plot(unique_ranks, cumsum_pct, linewidth=2, color='steelblue', marker='o', markersize=4)
        ax.axhline(50, color='red', linestyle='--', linewidth=2, label='50% of weight')
        ax.axhline(90, color='orange', linestyle='--', linewidth=2, label='90% of weight')
        ax.set_xlabel('Rank (1=best match)', fontsize=12)
        ax.set_ylabel('Cumulative Weight (%)', fontsize=12)
        ax.set_title('Cumulative Weight Distribution by Rank', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(unique_ranks) + 1)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        viz3_path = Path(output_dir) / 'weight_analysis.png'
        plt.savefig(viz3_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {viz3_path}")
    
    # =========================================================================
    # NEW VISUALIZATION 4: Preference Space Coverage
    # =========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract preferences
    parking_prefs = [s['preferences']['parking_importance'] for s in training_data]
    time_prefs = [s['preferences']['time_importance'] for s in training_data]
    dist_prefs = [s['preferences']['distance_importance'] for s in training_data]
    
    # Plot 1: 3D preference space
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    
    if has_weights:
        # Color by weight
        scatter = ax.scatter(parking_prefs, time_prefs, dist_prefs, 
                           c=weights, cmap='viridis', s=30, alpha=0.6)
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Sample Weight', fontsize=10)
    else:
        ax.scatter(parking_prefs, time_prefs, dist_prefs, 
                  c='steelblue', s=30, alpha=0.6)
    
    ax.set_xlabel('Parking Importance', fontsize=10)
    ax.set_ylabel('Time Importance', fontsize=10)
    ax.set_zlabel('Distance Importance', fontsize=10)
    ax.set_title('Preference Space Coverage', fontsize=12, fontweight='bold')
    
    # 2D projections of preference space
    projections = [
        (2, 2, 2, parking_prefs, time_prefs, 'Parking vs Time'),
        (2, 2, 3, parking_prefs, dist_prefs, 'Parking vs Distance'),
        (2, 2, 4, time_prefs, dist_prefs, 'Time vs Distance')
    ]
    
    for row, col, pos, x_data, y_data, title in projections:
        ax = fig.add_subplot(row, col, pos)
        
        if has_weights:
            scatter = ax.scatter(x_data, y_data, c=weights, cmap='viridis', s=30, alpha=0.6)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(x_data, y_data, c='steelblue', s=30, alpha=0.6)
        
        ax.set_xlabel(title.split(' vs ')[0] + ' Importance', fontsize=10)
        ax.set_ylabel(title.split(' vs ')[1] + ' Importance', fontsize=10)
        ax.set_title(f'Preference Space: {title}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    viz4_path = Path(output_dir) / 'preference_analysis.png'
    plt.savefig(viz4_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {viz4_path}")
    
    # =========================================================================
    # NEW VISUALIZATION 5: Cost Diversity Analysis
    # =========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract cost parameters
    cost_per_hour = [s['costs']['costPerTravelHour'] for s in training_data]
    cost_per_km = [s['costs']['costPerKm'] for s in training_data]
    parking_mult = [s['costs']['parking_multiplier'] for s in training_data]
    
    # Plot 1: Cost parameter distributions
    ax = axes[0, 0]
    ax.hist([cost_per_hour, cost_per_km, parking_mult], 
           bins=30, label=['costPerHour', 'costPerKm', 'parking_mult'],
           alpha=0.7, edgecolor='black')
    ax.set_xlabel('Cost Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Cost Parameter Distributions (Overlaid)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cost scatter - hour vs km
    ax = axes[0, 1]
    if has_weights:
        scatter = ax.scatter(cost_per_hour, cost_per_km, c=weights, cmap='viridis', s=30, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Sample Weight')
    else:
        ax.scatter(cost_per_hour, cost_per_km, c='steelblue', s=30, alpha=0.6)
    
    ax.set_xlabel('costPerTravelHour', fontsize=12)
    ax.set_ylabel('costPerKm', fontsize=12)
    ax.set_title('Cost Space: Hour vs Km', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cost scatter - hour vs parking
    ax = axes[1, 0]
    if has_weights:
        scatter = ax.scatter(cost_per_hour, parking_mult, c=weights, cmap='viridis', s=30, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Sample Weight')
    else:
        ax.scatter(cost_per_hour, parking_mult, c='steelblue', s=30, alpha=0.6)
    
    ax.set_xlabel('costPerTravelHour', fontsize=12)
    ax.set_ylabel('parking_multiplier', fontsize=12)
    ax.set_title('Cost Space: Hour vs Parking', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Unique cost combinations count
    ax = axes[1, 1]
    
    # Count unique cost combinations
    unique_costs = set(tuple(s['costs'].values()) for s in training_data)
    cost_counts = Counter(tuple(s['costs'].values()) for s in training_data)
    
    # Sort by frequency
    sorted_costs = sorted(cost_counts.items(), key=lambda x: x[1], reverse=True)
    top_n = min(20, len(sorted_costs))
    
    counts = [c[1] for c in sorted_costs[:top_n]]
    bars = ax.bar(range(top_n), counts, edgecolor='black', alpha=0.7, color='steelblue')
    
    ax.set_xlabel('Cost Combination Rank', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Top {top_n} Most Common Cost Combinations\n(Total unique: {len(unique_costs)})', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        pct = count / len(training_data) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
               f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    viz5_path = Path(output_dir) / 'cost_diversity.png'
    plt.savefig(viz5_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {viz5_path}")
    
    # =========================================================================
    # STATISTICS SUMMARY
    # =========================================================================
    
    stats_path = Path(output_dir) / 'statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TRAINING DATA STATISTICS (V7)\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Format: {'V7 (with sample weights)' if has_weights else 'V6 (no weights)'}\n\n")
        
        f.write(f"Cost Pool:\n")
        f.write(f"  Total solutions: {len(cost_pool)}\n")
        f.write(f"  Pareto front: {len(pareto_front)} ({len(pareto_front)/len(cost_pool)*100:.1f}%)\n\n")
        
        f.write(f"Training Data:\n")
        f.write(f"  Total samples: {len(training_data)}\n")
        f.write(f"  Unique solutions used: {len(training_pool_ids)}\n")
        f.write(f"  Avg samples per solution: {len(training_data)/len(training_pool_ids):.1f}\n")
        
        # Preference diversity
        unique_prefs = len(set(tuple(s['preferences'].values()) for s in training_data))
        f.write(f"  Unique preferences: {unique_prefs}\n")
        
        # Cost diversity
        unique_costs = len(set(tuple(s['costs'].values()) for s in training_data))
        f.write(f"  Unique cost combinations: {unique_costs}\n\n")
        
        if has_weights:
            weights = np.array([s['sample_weight'] for s in training_data])
            f.write(f"Sample Weights:\n")
            f.write(f"  Min: {weights.min():.3f} (worst match)\n")
            f.write(f"  Median: {np.median(weights):.3f}\n")
            f.write(f"  Max: {weights.max():.3f} (best match)\n")
            f.write(f"  Ratio: {weights.max() / weights.min():.1f}x\n")
            f.write(f"  Std dev: {np.std(weights):.3f}\n\n")
        
        f.write(f"Pareto Front Usage:\n")
        pareto_in_training = len([pid for pid in training_pool_ids if pid in pareto_ids])
        pareto_usage = sum(pool_id_counts.get(pid, 0) for pid in pareto_ids)
        f.write(f"  Pareto solutions in training: {pareto_in_training}/{len(pareto_front)}\n")
        f.write(f"  Pareto samples: {pareto_usage}/{len(training_data)} ({pareto_usage/len(training_data)*100:.1f}%)\n\n")
        
        f.write(f"Top 10 most used solutions:\n")
        for i, (pool_id, count) in enumerate(pool_id_counts.most_common(10), 1):
            is_pareto = "⭐ PARETO" if pool_id in pareto_ids else ""
            pct = count / len(training_data) * 100
            f.write(f"  #{i}. pool_id={pool_id}: {count} times ({pct:.1f}%) {is_pareto}\n")
        
        # Preference concentration analysis
        f.write(f"\nPreference Concentration:\n")
        pref_counts = Counter(tuple(s['preferences'].values()) for s in training_data)
        top_prefs = pref_counts.most_common(5)
        for i, (pref, count) in enumerate(top_prefs, 1):
            pct = count / len(training_data) * 100
            f.write(f"  #{i}. {pref}: {count} times ({pct:.1f}%)\n")
        
        # Cost concentration analysis
        f.write(f"\nCost Concentration:\n")
        cost_counts = Counter(tuple(s['costs'].values()) for s in training_data)
        top_costs = cost_counts.most_common(5)
        for i, (cost, count) in enumerate(top_costs, 1):
            pct = count / len(training_data) * 100
            f.write(f"  #{i}. {cost}: {count} times ({pct:.1f}%)\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"  ✓ Saved: {stats_path}")
    print(f"\n✅ All visualizations saved to: {output_dir}/\n")



