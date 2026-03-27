import serial, time, math
import torch, torch.nn as nn, torch.optim as optim
import numpy as np

PORT = '/dev/ttyS0' # Change to 'COM2' or 'COM3' if on Windows
BAUD = 115200 
BASELINE_FILE = 'motor_baseline.pt'

# --- 1. AI MODEL ARCHITECTURE ---
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

# --- 2. CALIBRATION ROUTINE ---
def run_calibration():
    print("=========================================")
    print(" PINN MOTOR DIGITAL TWIN CALIBRATION TOOL")
    print("=========================================")
    print(f"Listening on {PORT} for a Healthy Baseline Frame...")

    try:
        ser = serial.Serial(PORT, BAUD, timeout=2)
        ser.write(b't0510\r') # Request real sensor data
        time.sleep(0.5)
        ser.reset_input_buffer()
        
        raw_data = None
        current_friction = 0.0
        
        # 1. Capture exactly ONE valid frame
        while raw_data is None:
            if ser.in_waiting > 0:
                line = ser.read_until(b'\r').decode('utf-8', errors='ignore').strip()
                if line.startswith('t1004'):
                    temp = int(line[5:7], 16)
                    rms = int(line[7:9], 16)
                    current_friction = 1.00 - (temp * 0.0005)
                    
                    # Reconstruct the wave
                    wave, amp, decay = [], rms*1.414, current_friction
                    for i in range(50):
                        wave.append(int(amp * math.sin(i * 0.314)))
                        amp *= decay
                    raw_data = wave
                    print(f"[+] Frame Captured! Temp: {temp}C | RMS: {rms}")
        ser.close()

        # 2. Spin up the AI Engine
        print("[*] Engaging PINN Optimizer... Please wait.")
        ai_brain = PINN_Engine()
        optimizer = optim.Adam([{'params': ai_brain.net.parameters(), 'lr': 0.002},
                                {'params': [ai_brain.raw_c, ai_brain.raw_k], 'lr': 0.01}])
        
        ai_brain.train()
        t_data = torch.linspace(-1, 1, 50).view(-1, 1)
        x_data = torch.tensor(raw_data, dtype=torch.float32).view(-1, 1) / 512.0
        
        # 3. Deep Offline Training (250 Epochs)
        for epoch in range(250):
            optimizer.zero_grad()
            l_data = 100.0 * torch.mean((ai_brain(t_data) - x_data)**2)
            l_phys = ai_brain.loss(t_data)
            (l_data + l_phys).backward()
            torch.nn.utils.clip_grad_norm_(ai_brain.parameters(), 1.0)
            optimizer.step()
            
            if epoch % 50 == 0:
                print(f"    Epoch {epoch}/250 completed...")

        # 4. Save the Model
        torch.save({
            'model_state': ai_brain.state_dict(),
            'baseline_c': current_friction
        }, BASELINE_FILE)
        
        print("\n[SUCCESS] Golden Baseline Calibrated and Saved to disk!")
        print(f"[SUCCESS] Mechanical Friction Baseline: {current_friction:.4f}")
        
    except Exception as e:
        print(f"\n[ERROR] Calibration failed: {e}")

if __name__ == '__main__':
    run_calibration()