"""
Generate a visual diagram showing the data flow between different strategy systems
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import matplotlib.lines as mlines

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'Trading Bot Strategy Data Flow', fontsize=18, fontweight='bold', ha='center')

# Color scheme
color_external = '#FF6B6B'
color_database = '#4ECDC4'
color_sql = '#45B7D1'
color_executable = '#96CEB4'
color_bridge = '#FECA57'
color_arrow = '#555555'

# External Data Sources
external_box = FancyBboxPatch((1, 7.5), 12, 1, boxstyle="round,pad=0.1",
                              facecolor=color_external, edgecolor='black', linewidth=2)
ax.add_patch(external_box)
ax.text(7, 8, 'External Data Sources\n(Exchange APIs, WebSocket Feeds)', 
        ha='center', va='center', fontsize=11, fontweight='bold')

# Raw Data Collection
collection_box = FancyBboxPatch((4, 6), 6, 0.8, boxstyle="round,pad=0.1",
                                facecolor='#FFE5B4', edgecolor='black', linewidth=2)
ax.add_patch(collection_box)
ax.text(7, 6.4, 'Raw Data Collection Pipeline', ha='center', va='center', fontsize=10)

# SQLite Database
db_box = FancyBboxPatch((3, 3.5), 8, 1.5, boxstyle="round,pad=0.1",
                        facecolor=color_database, edgecolor='black', linewidth=2)
ax.add_patch(db_box)
ax.text(7, 4.7, 'SQLite Database', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(7, 4.2, 'Tables:', ha='center', va='center', fontsize=9)
ax.text(7, 3.9, 'ohlcv_data | indicators | rsi_data | macd_data', 
        ha='center', va='center', fontsize=8)
ax.text(7, 3.6, 'bollinger_bands_data | signals | trades', 
        ha='center', va='center', fontsize=8)

# SQL-based Descriptors
sql_box = FancyBboxPatch((0.5, 1), 5, 1.8, boxstyle="round,pad=0.1",
                         facecolor=color_sql, edgecolor='black', linewidth=2)
ax.add_patch(sql_box)
ax.text(3, 2.3, 'SQL-based Descriptors', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(3, 1.9, '• Historical Analysis', ha='center', va='center', fontsize=9)
ax.text(3, 1.6, '• Pre-calculated Indicators', ha='center', va='center', fontsize=9)
ax.text(3, 1.3, '• Backtesting Queries', ha='center', va='center', fontsize=9)

# Executable Strategies
exec_box = FancyBboxPatch((8.5, 1), 5, 1.8, boxstyle="round,pad=0.1",
                          facecolor=color_executable, edgecolor='black', linewidth=2)
ax.add_patch(exec_box)
ax.text(11, 2.3, 'Executable Strategies', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(11, 1.9, '• Real-time Calculations', ha='center', va='center', fontsize=9)
ax.text(11, 1.6, '• Live Signal Generation', ha='center', va='center', fontsize=9)
ax.text(11, 1.3, '• Dynamic Adaptation', ha='center', va='center', fontsize=9)

# Strategy Bridge
bridge_box = FancyBboxPatch((5.5, 0.2), 3, 0.6, boxstyle="round,pad=0.1",
                            facecolor=color_bridge, edgecolor='black', linewidth=2)
ax.add_patch(bridge_box)
ax.text(7, 0.5, 'Strategy Bridge', ha='center', va='center', fontsize=10, fontweight='bold')

# Arrows
# External to Collection
arrow1 = FancyArrowPatch((7, 7.5), (7, 6.8), 
                         connectionstyle="arc3", arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color=color_arrow)
ax.add_patch(arrow1)

# Collection to Database
arrow2 = FancyArrowPatch((7, 6), (7, 5), 
                         connectionstyle="arc3", arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color=color_arrow)
ax.add_patch(arrow2)

# Database to SQL Descriptors
arrow3 = FancyArrowPatch((5, 3.5), (3, 2.8), 
                         connectionstyle="arc3,rad=0.3", arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color=color_arrow)
ax.add_patch(arrow3)

# Database to Executable (optional)
arrow4 = FancyArrowPatch((9, 3.5), (11, 2.8), 
                         connectionstyle="arc3,rad=-0.3", arrowstyle='->', 
                         mutation_scale=20, linewidth=1.5, color=color_arrow, 
                         linestyle='dashed')
ax.add_patch(arrow4)

# Direct data to Executable
arrow5 = FancyArrowPatch((10, 6), (11, 2.8), 
                         connectionstyle="arc3,rad=-0.3", arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color=color_arrow)
ax.add_patch(arrow5)
ax.text(11.5, 4, 'Live Data', ha='center', va='center', fontsize=9, 
        rotation=-60, color=color_arrow)

# Bridge connections
arrow6 = FancyArrowPatch((3, 1), (6.5, 0.8), 
                         connectionstyle="arc3,rad=0.3", arrowstyle='->', 
                         mutation_scale=15, linewidth=1.5, color=color_arrow)
ax.add_patch(arrow6)

arrow7 = FancyArrowPatch((11, 1), (7.5, 0.8), 
                         connectionstyle="arc3,rad=-0.3", arrowstyle='->', 
                         mutation_scale=15, linewidth=1.5, color=color_arrow)
ax.add_patch(arrow7)

# Add labels for data types
ax.text(5.5, 5.5, 'OHLCV\nData', ha='center', va='center', fontsize=8, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray'))

ax.text(2, 3.2, 'SQL\nQueries', ha='center', va='center', fontsize=8, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray'))

ax.text(12, 3.2, 'DataFrame', ha='center', va='center', fontsize=8, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray'))

# Legend
legend_elements = [
    mlines.Line2D([0], [0], color=color_arrow, lw=2, label='Primary Data Flow'),
    mlines.Line2D([0], [0], color=color_arrow, lw=2, linestyle='--', label='Optional Data Flow'),
    patches.Patch(facecolor=color_sql, label='Historical Analysis'),
    patches.Patch(facecolor=color_executable, label='Real-time Trading'),
    patches.Patch(facecolor=color_bridge, label='Integration Layer')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 0.95))

# Add notes
note_text = ("Note: Executable strategies can operate independently of the database,\n"
             "calculating indicators on-the-fly from live OHLCV data")
ax.text(7, 0, note_text, ha='center', va='center', fontsize=8, 
        style='italic', color='#666666')

plt.tight_layout()
plt.savefig('strategy_data_flow_diagram.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()
