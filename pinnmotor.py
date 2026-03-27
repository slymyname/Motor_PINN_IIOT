import sys, serial, time, csv, math, os
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg

PORT = '/dev/ttyS0' 
BAUD = 115200 
BASELINE_FILE = 'motor_baseline.pt'

# ==============================================================================
# 1. THE SIREN ENGINE (AI Model)
# ==============================================================================
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
        self.raw_c = nn.Parameter(torch.tensor([0.0], requires_grad=True))
        self.raw_k = nn.Parameter(torch.tensor([0.0], requires_grad=True))
    def get_params(self):
        return torch.sigmoid(self.raw_c)*2.0, 1.0+(torch.sigmoid(self.raw_k)*4.0)
    def forward(self, t): return self.net(t)
    def loss(self, t):
        t.requires_grad = True
        x = self.forward(t)
        dx_dt = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
        d2x_dt2 = torch.autograd.grad(dx_dt, t, torch.ones_like(dx_dt), create_graph=True)[0]
        c, k = self.get_params()
        return torch.mean((d2x_dt2 + c*dx_dt + k*50*x)**2)

ai_brain = PINN_Engine()
optimizer = optim.Adam([{'params': ai_brain.net.parameters(), 'lr': 0.002},
                        {'params': [ai_brain.raw_c, ai_brain.raw_k], 'lr': 0.01}])

def estimate_friction(samples):
    t_data = torch.linspace(-1, 1, len(samples)).view(-1, 1)
    x_data = torch.tensor(samples, dtype=torch.float32).view(-1, 1) / 512.0
    for _ in range(40): 
        optimizer.zero_grad()
        l_data = 100.0 * torch.mean((ai_brain(t_data) - x_data)**2)
        l_phys = ai_brain.loss(t_data)
        (l_data + l_phys).backward()
        torch.nn.utils.clip_grad_norm_(ai_brain.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        curve = ai_brain(t_data).numpy().flatten() * 512.0 
        c, k = ai_brain.get_params()
    return c.item(), curve.tolist()

# ==============================================================================
# 2. SLCAN BACKGROUND THREAD 
# ==============================================================================
class SerialWorker(QThread):
    data_received = pyqtSignal(dict)
    
    def run(self):
        try:
            ser = serial.Serial(PORT, BAUD, timeout=1)
            ser.write(b't0510\r') 
            time.sleep(0.5)
            ser.reset_input_buffer()
            
            while True:
                if ser.in_waiting > 0:
                    raw_bytes = ser.read_until(b'\r')
                    line = raw_bytes.decode('utf-8', errors='ignore').strip()
                    
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
                                    "Status": st_txt, "Data": wave, "CrestFactor": 1.41}
                            
                            c, curve = estimate_friction(data["Data"])
                            data["Friction"], data["FittedCurve"] = c, curve
                            
                            self.data_received.emit(data)
                            ser.reset_input_buffer() 
                            
                        except Exception as inner_err:
                            print(f"Data Processing Error: {inner_err}")
        except Exception as e: 
            print(f"Fatal Serial Error: {e}")

# ==============================================================================
# 3. DASHBOARD GUI & CSV LOGGER
# ==============================================================================
class MotorDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MOTOR_PINN_IIoT - Predictive Maintenance")
        self.setGeometry(100, 100, 1250, 750) 
        self.setStyleSheet("background-color: #1e1e2e; color: white;")
        
        self.baseline_c = None
        self.current_friction = 0.0
        
        # Setup Logger
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_file = open(f"motor_log_CAN_{ts}.csv", mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Time", "RMS", "Temp", "SpeedPct", "Friction", "HealthPct"])
        
        self.initUI()
        self.worker = SerialWorker()
        self.worker.data_received.connect(self.update_ui)
        self.worker.start()

    def initUI(self):
        layout = QVBoxLayout()
        h = QLabel("MOTOR HEALTH STATUS (PREDICTIVE MAINTENANCE)")
        h.setFont(QFont('Arial', 18, QFont.Bold)); h.setAlignment(Qt.AlignCenter); layout.addWidget(h)

        # Metrics Row
        row = QHBoxLayout()
        self.health_lbl = self.mk_box("Health Score (%)", row, "#89b4fa")
        self.rms_lbl = self.mk_box("Vibration (RMS)", row)
        self.temp_lbl = self.mk_box("Motor Temp (C)", row, "#a6e3a1") 
        self.fric_lbl = self.mk_box("AI Damping (c)", row, "#f38ba8")
        self.motor_lbl = self.mk_box("Motor Speed (%)", row, "#a6e3a1")
        layout.addLayout(row)

        # Baseline Controls Row
        btn_row = QHBoxLayout()
        self.btn_save = QPushButton("💾 Save Current State as Baseline")
        self.btn_load = QPushButton("📂 Load Last Baseline")
        
        btn_style = "font-weight: bold; padding: 10px; border-radius: 5px; color: black;"
        self.btn_save.setStyleSheet(btn_style + "background-color: #a6e3a1;")
        self.btn_load.setStyleSheet(btn_style + "background-color: #89b4fa;")
        
        self.btn_save.clicked.connect(self.save_baseline)
        self.btn_load.clicked.connect(self.load_baseline)
        
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_load)
        layout.addLayout(btn_row)

        # Graph
        self.plot = pg.PlotWidget(title="Real-Time Digital Twin")
        self.plot.setBackground('#313244')
        self.raw_curve = self.plot.plot(pen=pg.mkPen('#89b4fa', width=2))
        self.ai_curve = self.plot.plot(pen=pg.mkPen('#f38ba8', width=3, style=Qt.DashLine))
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

    def save_baseline(self):
        """Saves the AI weights and current friction to establish 'Perfect Health'"""
        if self.current_friction > 0:
            torch.save({
                'model_state': ai_brain.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'baseline_c': self.current_friction
            }, BASELINE_FILE)
            self.baseline_c = self.current_friction
            self.stat_lbl.setText(f"Golden Record Saved! Baseline Friction: {self.baseline_c:.4f}")

    def load_baseline(self):
        """Loads the saved 'Perfect Health' profile"""
        if os.path.exists(BASELINE_FILE):
            checkpoint = torch.load(BASELINE_FILE)
            ai_brain.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.baseline_c = checkpoint['baseline_c']
            self.stat_lbl.setText(f"Golden Record Loaded! Baseline Friction: {self.baseline_c:.4f}")
        else:
            self.stat_lbl.setText("No baseline file found. Please save one first!")

    def update_ui(self, data):
        t, r, f, p = data['Temp'], data['RMS'], data['Friction'], data['MotorPWM']
        pct = int((p/255.0)*100)
        self.current_friction = f
        
        # --- TUNED HEALTH SCORE CALCULATION ---
        health = 100.0
        if self.baseline_c is not None and self.baseline_c > 0:
            # 1. Calculate raw deviation
            deviation = (f - self.baseline_c) / self.baseline_c
            
            # 2. TUNING PARAMETERS
            sensitivity = 0.5  # Lower = less sensitive. Try 0.2 or 0.5.
            deadzone = 0.05    # Ignore the first 5% of friction increase
            
            if deviation > deadzone:
                # Apply penalty only after crossing the deadzone
                adjusted_deviation = (deviation - deadzone) * sensitivity
                health = max(0.0, 100.0 - (adjusted_deviation * 100.0))
            else:
                health = 100.0 # Everything is fine within the deadzone
            
            # Update the UI
            self.health_lbl.setText(f"{health:.1f}%")
            if health < 50: self.health_lbl.setStyleSheet("color: #f38ba8;") # Red
            elif health < 85: self.health_lbl.setStyleSheet("color: #f9e2af;") # Yellow
            else: self.health_lbl.setStyleSheet("color: #a6e3a1;") # Green
        else:
            self.health_lbl.setText("N/A")
            self.health_lbl.setStyleSheet("color: #89b4fa;")
        
        self.rms_lbl.setText(f"{r:.2f}")
        self.temp_lbl.setText(f"{t}"); self.fric_lbl.setText(f"{f:.4f}")
        self.motor_lbl.setText(f"{pct}%")
        
        self.temp_lbl.setStyleSheet(f"color: {'#f38ba8' if t>=85 else '#f9e2af' if t>=60 else '#a6e3a1'};")
        self.motor_lbl.setStyleSheet(f"color: {'#a6e3a1' if pct==100 else '#f9e2af' if pct>0 else '#f38ba8'};")
        
        self.raw_curve.setData(data["Data"])
        self.ai_curve.setData(data["FittedCurve"])
        
        self.csv_writer.writerow([datetime.now().strftime('%H:%M:%S.%f')[:-3], r, t, pct, f, health])
        
        if "Loaded" not in self.stat_lbl.text() and "Saved" not in self.stat_lbl.text():
            st = data["Status"]
            self.stat_lbl.setText(st)
            self.stat_lbl.setStyleSheet(f"background-color: {'#f38ba8' if 'CRIT' in st else '#a6e3a1'}; color: black; padding: 10px;")

    def closeEvent(self, event):
        print("Closing application and saving CSV log...")
        self.csv_file.close()
        self.worker.terminate()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv); app.setStyle("Fusion")
    win = MotorDashboard(); win.show(); sys.exit(app.exec_())