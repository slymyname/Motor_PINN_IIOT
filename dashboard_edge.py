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

def run_inference(temp):
    t_data = torch.linspace(-1, 1, 50).view(-1, 1)
    with torch.no_grad(): # MAXIMUM SPEED
        golden_curve = ai_brain(t_data).numpy().flatten() * 512.0 
    live_friction = 1.00 - (temp * 0.0005) 
    return live_friction, golden_curve.tolist()

# --- 2. SLCAN BACKGROUND THREAD ---
class SerialWorker(QThread):
    data_received = pyqtSignal(dict)
    def run(self):
        try:
            ser = serial.Serial(PORT, BAUD, timeout=1)
            ser.reset_input_buffer()
            while True:
                if ser.in_waiting > 0:
                    line = ser.read_until(b'\r').decode('utf-8', errors='ignore').strip()
                    if line.startswith('t1004') and len(line) >= 13:
                        try:
                            temp = int(line[5:7], 16)
                            rms = int(line[7:9], 16)
                            pwm = int(line[9:11], 16)
                            st_code = int(line[11:13], 16)
                            
                            st_txt = "CRITICAL_ESTOP" if st_code==2 else "Warning_Throttled" if st_code==1 else "Healthy"
                            
                            wave, amp, decay = [], rms*1.414, 1.00-(temp*0.0005)
                            for i in range(50):
                                wave.append(int(amp * math.sin(i * 0.314)))
                                amp *= decay
                                
                            data = {"Temp": temp, "RMS": rms, "MotorPWM": pwm, 
                                    "Status": st_txt, "Data": wave}
                            
                            c, curve = run_inference(temp)
                            data["Friction"], data["FittedCurve"] = c, curve
                            self.data_received.emit(data)
                            ser.reset_input_buffer() 
                        except Exception: pass
        except Exception as e: print(f"Serial Error: {e}")

# --- 3. DASHBOARD GUI ---
class MotorDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PINN Motor IIoT - Edge Inference UI")
        self.setGeometry(100, 100, 1250, 750) 
        self.setStyleSheet("background-color: #1e1e2e; color: white;")
        
        self.baseline_c = None
        self.initUI()
        self.load_baseline() # Load weights automatically on boot
        
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
            ai_brain.eval() # STRICTLY LOCK WEIGHTS
            self.stat_lbl.setText(f"Golden Record Active! Baseline Friction: {self.baseline_c:.4f}")
            self.stat_lbl.setStyleSheet("background-color: #a6e3a1; color: black; padding: 10px;")
        else:
            self.stat_lbl.setText("NO BASELINE FOUND! Please run calibrate_twin.py first.")
            self.stat_lbl.setStyleSheet("background-color: #f38ba8; color: black; padding: 10px;")

    def update_ui(self, data):
        t, r, f, p = data['Temp'], data['RMS'], data['Friction'], data['MotorPWM']
        pct = int((p/255.0)*100)
        
        health = 100.0
        if self.baseline_c is not None and self.baseline_c > 0:
            # THE FIX: Absolute value catches both positive and negative deviation
            deviation = abs(f - self.baseline_c) / self.baseline_c
            
            # TWEAKED: Lowered deadzone to 2% (0.02) to react faster to heat spikes
            sensitivity, deadzone = 30, 0.02    
            
            if deviation > deadzone:
                health = max(0.0, 100.0 - ((deviation - deadzone) * sensitivity * 100.0))
            
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
        
        if "Active" not in self.stat_lbl.text():
            st = data["Status"]
            self.stat_lbl.setText(st)
            self.stat_lbl.setStyleSheet(f"background-color: {'#f38ba8' if 'CRIT' in st else '#a6e3a1'}; color: black; padding: 10px;")

    def closeEvent(self, event):
        self.worker.terminate()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv); app.setStyle("Fusion")
    win = MotorDashboard(); win.show(); sys.exit(app.exec_())