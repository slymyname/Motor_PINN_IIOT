import sys, serial, time, csv, math, os
import numpy as np
import torch, torch.nn as nn
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg

PORT = '/dev/ttyS0' # Change to COM port if necessary
BAUD = 115200 
BASELINE_FILE = 'motor_baseline.pt'

# --- 1. AI MODEL ARCHITECTURE (Required for Loading) ---
class SIREN_Layer(nn.Module):
    def __init__(self, in_f, out_f, is_first=False):
        super().__init__()
        self.w0 = 15.0 
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            if is_first: self.linear.weight.uniform_(-1/in_f, 1/in_f)
            else: self.linear.weight.uniform_(-np.sqrt(6/in_f)/self.w0, np.sqrt(6/in_f)/self.w0)
    def forward(self, x): return torch.sin(self.w0 * self.linear(x))

class PINN_Engine(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(SIREN_Layer(1, 64, True), SIREN_Layer(64, 64), nn.Linear(64, 1))
        self.raw_c = nn.Parameter(torch.tensor([0.0]))
        self.raw_k = nn.Parameter(torch.tensor([0.0]))
    def forward(self, t): return self.net(t)

# --- GLOBAL INFERENCE ENGINE ---
ai_brain = PINN_Engine()
# Note: No optimizer here! Pure inference.

def run_pinn_diagnostics(live_rms, live_wave):
    """
    A TRUE Edge AI Inference Step.
    Compares the physical sensor reality against the PINN's thermodynamic expectations.
    """
    t_data = torch.linspace(-1, 1, 50).view(-1, 1)
    with torch.no_grad():
        expected_peak = live_rms * 1.414 
        golden_curve = (ai_brain(t_data).numpy().flatten() * expected_peak).tolist()

    error_sum = 0
    for i in range(50):
        error_sum += abs(golden_curve[i] - live_wave[i])
    
    # Calculate raw error
    raw_mae = error_sum / 50.0

    # Normalize the error by the power level
    # Prevent division by zero if motor is completely stopped
    safe_peak = max(expected_peak, 1.0) 
    
    # This gives us a percentage of distortion (e.g., 0.05 = 5% distortion)
    normalized_error = raw_mae / safe_peak 

    # Derive Health from the RELATIVE distortion. 
    health_score = max(0.0, 100.0 - (normalized_error * 300.0)) 
    
    return health_score, golden_curve

# --- 2. SLCAN BACKGROUND THREAD WITH AUTO-LOGGING ---
class SerialWorker(QThread):
    data_received = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.smoothed_health = 100.0  
        self.alpha = 0.15  
        self._is_running = True 

    def run(self):
        # 1. Initialize the CSV Logger
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'motor_log_{timestamp_str}.csv'
        print(f"[*] Auto-Logging started: Saving telemetry to {log_filename}")

        try:
            # Open file in append/write mode
            with open(log_filename, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                # Write standard headers
                csv_writer.writerow(['Time_Elapsed', 'Temperature', 'RMS_Current', 'PWM', 'Live_Damping', 'Health_Score', 'Status'])
                
                ser = serial.Serial(PORT, BAUD, timeout=1)
                ser.reset_input_buffer()
                start_time = time.time()
                
                while self._is_running:
                    if ser.in_waiting > 0:
                        line = ser.read_until(b'\r').decode('utf-8', errors='ignore').strip()
                        if line.startswith('t1004') and len(line) >= 13:
                            try:
                                temp = int(line[5:7], 16)
                                rms = int(line[7:9], 16)
                                pwm = int(line[9:11], 16)
                                st_code = int(line[11:13], 16)
                                
                                st_txt = "CRITICAL_ESTOP" if st_code==2 else "Warning_Throttled" if st_code==1 else "Healthy"
                                
                                live_wave = []
                                amp = rms * 1.414
                                decay = 1.00 - (temp * 0.0005) 
                                for i in range(50):
                                    live_wave.append(int(amp * math.sin(i * 0.314)))
                                    amp *= decay 
                                    
                                # Get the raw inference
                                raw_ai_health, golden_wave = run_pinn_diagnostics(rms, live_wave)
                                
                                # Apply Mechanical Inertia (EMA)
                                self.smoothed_health = (self.alpha * raw_ai_health) + ((1.0 - self.alpha) * self.smoothed_health)
                                
                                # --- THE FIX: STATUS OVERRIDE ---
                                # If the RTOS has killed the motor due to critical heat, 
                                # override the vibration math and force health to 0.
                                if st_code == 2: # 2 is the code for CRITICAL_ESTOP
                                    self.smoothed_health = 0.0
                                
                                # 2. WRITE ROW TO CSV 
                                elapsed = time.time() - start_time
                                csv_writer.writerow([f"{elapsed:.2f}", temp, rms, pwm, f"{decay:.4f}", f"{self.smoothed_health:.1f}", st_txt])
                                csv_file.flush() 
                                
                                data = {
                                    "Temp": temp, 
                                    "RMS": rms, 
                                    "MotorPWM": pwm, 
                                    "Status": st_txt, 
                                    "Data": live_wave,
                                    "FittedCurve": golden_wave,
                                    "TrueHealth": self.smoothed_health, 
                                    "LiveDamping": decay
                                }
                                self.data_received.emit(data)
                            except Exception: pass
                
                # Close serial port gracefully when loop exits
                ser.close()
                print("[*] Serial port closed.")

        except Exception as e: print(f"Serial Error: {e}")

    def stop(self):
        self._is_running = False

# --- 3. DASHBOARD GUI ---
class MotorDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PINN Motor IIoT - Edge Inference UI")
        self.setGeometry(100, 100, 1250, 750) 
        self.setStyleSheet("background-color: #1e1e2e; color: white;")
        
        self.baseline_c = None
        self.initUI()
        self.load_baseline() 
        
        self.worker = SerialWorker()
        self.worker.data_received.connect(self.update_ui)
        self.worker.start()

    def initUI(self):
        layout = QVBoxLayout()
        h = QLabel("MOTOR HEALTH DASHBOARD (INFERENCE MODE)")
        h.setFont(QFont('Arial', 18, QFont.Bold)); h.setAlignment(Qt.AlignCenter); layout.addWidget(h)

        row = QHBoxLayout()
        self.health_lbl = self.mk_box("Health Score (%)", row, "#89b4fa")
        self.rms_lbl = self.mk_box("Vibration (RMS)", row)
        self.temp_lbl = self.mk_box("Motor Temp (C)", row, "#a6e3a1") 
        self.fric_lbl = self.mk_box("Live Damping (c)", row, "#f38ba8")
        self.motor_lbl = self.mk_box("Motor Speed (%)", row, "#a6e3a1")
        layout.addLayout(row)

        self.btn_load = QPushButton("🔄 Reload Golden Baseline File")
        self.btn_load.setStyleSheet("font-weight: bold; padding: 10px; border-radius: 5px; color: black; background-color: #89b4fa;")
        self.btn_load.clicked.connect(self.load_baseline)
        layout.addWidget(self.btn_load)

        self.plot = pg.PlotWidget(title="Real-Time Digital Twin")
        self.plot.setBackground('#313244')
        
        # Lock X-axis to the 50-sample range, disable mouse panning for stability
        self.plot.setXRange(0, 49, padding=0)
        self.plot.setMouseEnabled(x=False, y=True) 
        self.plot.enableAutoRange(axis=pg.ViewBox.YAxis) 
        
        # Set some padding so the wave doesn't touch the absolute top/bottom edges
        self.plot.getViewBox().setDefaultPadding(0.1)

        self.raw_curve = self.plot.plot(pen=pg.mkPen('#89b4fa', width=2), name="Live Sensor")
        self.ai_curve = self.plot.plot(pen=pg.mkPen('#f38ba8', width=3, style=Qt.DashLine), name="Golden Baseline")
        layout.addWidget(self.plot)
        
        self.stat_lbl = QLabel("Awaiting CAN Data..."); self.stat_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.stat_lbl)
        
        c = QWidget(); c.setLayout(layout); self.setCentralWidget(c)

    def mk_box(self, title, layout, color="#89b4fa"):
        f = QFrame(); f.setStyleSheet("background-color: #313244; border-radius: 10px;")
        l = QVBoxLayout(); f.setLayout(l)
        t = QLabel(title); t.setAlignment(Qt.AlignCenter); l.addWidget(t)
        v = QLabel("N/A"); v.setFont(QFont('Arial', 24, QFont.Bold))
        v.setAlignment(Qt.AlignCenter); v.setStyleSheet(f"color: {color};"); l.addWidget(v)
        layout.addWidget(f); return v

    def load_baseline(self):
        if os.path.exists(BASELINE_FILE):
            checkpoint = torch.load(BASELINE_FILE)
            ai_brain.load_state_dict(checkpoint['model_state'])
            self.baseline_c = checkpoint['baseline_c']
            ai_brain.eval() 
            self.stat_lbl.setText(f"Golden Record Active! Baseline Friction: {self.baseline_c:.4f}")
            self.stat_lbl.setStyleSheet("background-color: #a6e3a1; color: black; padding: 10px;")
        else:
            self.stat_lbl.setText("NO BASELINE FOUND! Please run calibrate_twin.py first.")
            self.stat_lbl.setStyleSheet("background-color: #f38ba8; color: black; padding: 10px;")

    def update_ui(self, data):
        t, r, f, p = data['Temp'], data['RMS'], data['LiveDamping'], data['MotorPWM']
        pct = int((p/255.0)*100)
        health = data['TrueHealth']
            
        self.health_lbl.setText(f"{health:.1f}%")
        if health < 50: self.health_lbl.setStyleSheet("color: #f38ba8;") 
        elif health < 85: self.health_lbl.setStyleSheet("color: #f9e2af;") 
        else: self.health_lbl.setStyleSheet("color: #a6e3a1;") 
        
        self.rms_lbl.setText(f"{r:.2f}")
        self.temp_lbl.setText(f"{t}"); self.fric_lbl.setText(f"{f:.4f}")
        self.motor_lbl.setText(f"{pct}%")
        
        self.temp_lbl.setStyleSheet(f"color: {'#f38ba8' if t>=85 else '#f9e2af' if t>=60 else '#a6e3a1'};")
        self.motor_lbl.setStyleSheet(f"color: {'#a6e3a1' if pct==100 else '#f9e2af' if pct>0 else '#f38ba8'};")
        
        self.raw_curve.setData(data["Data"])
        self.ai_curve.setData(data["FittedCurve"])
        
        # Force the plot ViewBox to re-evaluate the auto-range bounds on every new frame
        self.plot.getViewBox().autoRange()
        
        if "Active" not in self.stat_lbl.text():
            st = data["Status"]
            self.stat_lbl.setText(st)
            self.stat_lbl.setStyleSheet(f"background-color: {'#f38ba8' if 'CRIT' in st else '#a6e3a1'}; color: black; padding: 10px;")

    def closeEvent(self, event):
        print("[*] Shutting down UI and background threads...")
        self.worker.stop()
        self.worker.wait() 
        event.accept()
        QApplication.quit() 

if __name__ == '__main__':
    app = QApplication(sys.argv); app.setStyle("Fusion")
    win = MotorDashboard(); win.show(); sys.exit(app.exec_())