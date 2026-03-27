import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import sys
import os

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else 'motor_log_20260327_065615.csv'

def generate_post_mortem_report(filename):
    if not os.path.exists(filename):
        print(f"Error: Could not find {filename}")
        return

    df = pd.read_csv(filename)
    x = df['Time_Elapsed']

    # 1. Setup Figure with GridSpec (Ratio 4:1 for Graph vs Legend space)
    fig = plt.figure(figsize=(15, 14), facecolor='white')
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.3, right=0.82) 
    
    # 2. Add Left-Aligned Main Title
    fig.text(0.08, 0.96, 'Digital Twin Analytics', 
             fontsize=24, fontweight='bold', color='#1a1a1a')
    fig.text(0.08, 0.94, f' Analysis | Source: {filename}', 
             fontsize=12, color='gray')

    # --- PANEL 1: HARDWARE ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_t = ax1.twinx()
    p1 = ax1.plot(x, df['Temperature'], color='tab:red', lw=2, label='Temp (°C)')
    h1 = ax1.axhline(90, color='red', ls='--', alpha=0.4, label='E-Stop (90°C)')
    h2 = ax1.axhline(75, color='orange', ls='--', alpha=0.4, label='Throttling (75°C)')
    p2 = ax1_t.step(x, df['PWM'], color='tab:blue', where='post', lw=2, label='PWM Out')
    
    ax1.set_title('1. HARDWARE TELEMETRY', fontsize=12, fontweight='bold', loc='left', color='#333333')
    ax1.set_ylabel('Temp (°C)', color='tab:red', fontweight='bold')
    ax1_t.set_ylabel('PWM (0-255)', color='tab:blue', fontweight='bold')
    
    # Legend 1 (Placed at 1.05 to be close but not touching)
    all_h1 = p1 + [h1, h2] + p2
    ax1.legend(handles=all_h1, labels=[l.get_label() for l in all_h1], 
               loc='center left', bbox_to_anchor=(1.08, 0.5), frameon=True, edgecolor='#cccccc')

    # --- PANEL 2: PHYSICS ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2_t = ax2.twinx()
    p3 = ax2.plot(x, df['Live_Damping'], color='tab:green', lw=2, label='Damping (c)')
    h3 = ax2.axhline(1.00 - (25 * 0.0005), color='darkgreen', ls=':', label='Baseline (25°C)')
    p4 = ax2_t.plot(x, df['RMS_Current'], color='tab:orange', lw=2, alpha=0.7, label='Vibration (RMS)')
    
    ax2.set_title('2. MECHANICAL PHYSICS (PINN)', fontsize=12, fontweight='bold', loc='left', color='#333333')
    ax2.set_ylabel('Damping (c)', color='tab:green', fontweight='bold')
    ax2_t.set_ylabel('Current (RMS)', color='tab:orange', fontweight='bold')
    
    all_h2 = p3 + [h3] + p4
    ax2.legend(handles=all_h2, labels=[l.get_label() for l in all_h2], 
               loc='center left', bbox_to_anchor=(1.08, 0.5), frameon=True, edgecolor='#cccccc')

    # --- PANEL 3: AI INFERENCE ---
    ax3 = fig.add_subplot(gs[2, 0])
    p5 = ax3.plot(x, df['Health_Score'], color='tab:purple', lw=3, label='Health %')
    ax3.fill_between(x, df['Health_Score'], color='tab:purple', alpha=0.1)
    
    # Status Backgrounds
    status_changes = df['Status'].ne(df['Status'].shift()).cumsum()
    for _, group in df.groupby(status_changes):
        st = group['Status'].iloc[0]
        c = '#ffcccc' if 'CRIT' in st else '#fff4cc' if 'Warning' in st else '#ccffcc'
        ax3.axvspan(group['Time_Elapsed'].iloc[0], group['Time_Elapsed'].iloc[-1], color=c, alpha=0.4)

    ax3.set_title('3. EDGE AI INFERENCE OUTPUT', fontsize=12, fontweight='bold', loc='left', color='#333333')
    ax3.set_ylabel('Score (%)', color='tab:purple', fontweight='bold')
    ax3.set_xlabel('Mission Time (Seconds)', fontsize=10)
    ax3.set_ylim(-5, 110)

    # Status Patches
    patches = [Patch(fc='#ccffcc', label='Healthy State'),
               Patch(fc='#fff4cc', label='Throttled State'),
               Patch(fc='#ffcccc', label='E-STOP State')]
    
    all_h3 = p5 + patches
    ax3.legend(handles=all_h3, labels=[l.get_label() for l in all_h3], 
               loc='center left', bbox_to_anchor=(1.08, 0.5), frameon=True, edgecolor='#cccccc')

    # Global cleanup
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, which='both', ls='-', alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.savefig(filename.replace('.csv', '_architectural_report.png'), dpi=300, bbox_inches='tight')
    print("[+] Report exported with architectural alignment.")
    plt.show()

if __name__ == "__main__":
    generate_post_mortem_report(LOG_FILE)