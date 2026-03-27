import serial
import time
import pytest

# --- LINUX CAN BRIDGE CONFIGURATION ---
PORT = '/dev/ttyS0' 
BAUD = 115200

@pytest.fixture(scope="module")
def motor_bus():
    """Sets up the Serial Bus and ensures it resets after testing."""
    ser = serial.Serial(PORT, BAUD, timeout=1)
    
    # Send SLCAN Reset Command (ID: 0x051, DLC: 0) to ensure a clean state
    ser.write(b't0510\r')
    time.sleep(1) 
    ser.reset_input_buffer()
    
    yield ser
    
    # --- TEARDOWN PHASE ---
    print("\nTests complete. Reverting Arduino to Physical Sensors...")
    ser.write(b't0510\r') # Send Reset Command again
    time.sleep(0.5)
    ser.close()

def get_clean_packet(ser):
    """Reads and decodes a single SLCAN Telemetry Frame from the Arduino."""
    ser.reset_input_buffer()
    time.sleep(0.5) # Wait for a fresh frame to arrive
    
    for _ in range(15):
        # Read until Carriage Return (SLCAN standard)
        raw_bytes = ser.read_until(b'\r')
        line = raw_bytes.decode('utf-8', errors='ignore').strip()
        
        # Look for Telemetry Frame: ID 0x100, Length 4
        if line.startswith('t1004') and len(line) >= 13:
            try:
                temp = int(line[5:7], 16)
                rms = int(line[7:9], 16)
                pwm = int(line[9:11], 16)
                status_code = int(line[11:13], 16)
                
                return {
                    "Temp": temp, 
                    "RMS": rms, 
                    "MotorPWM": pwm, 
                    "Status": status_code
                }
            except ValueError:
                continue
    return None

def test_autonomous_throttle_down(motor_bus):
    # Command: Set Temp to 70C. 
    # SLCAN format: t + ID(050) + Length(1) + Data(46) + \r  (70 in hex is 46)
    print("\n[TEST] Injecting 70C Warning Temperature...")
    motor_bus.write(b't050146\r')
    time.sleep(1.0) # Give Arduino time to process and adjust PWM
    
    data = get_clean_packet(motor_bus)
    assert data is not None, "Failed to receive CAN frame from Arduino"
    assert data["MotorPWM"] == 128, f"Expected PWM 128, but got {data['MotorPWM']}"
    assert data["Temp"] == 70, f"Arduino did not acknowledge Temp override"

def test_emergency_stop(motor_bus):
    # Command: Set Temp to 95C. 
    # SLCAN format: t + ID(050) + Length(1) + Data(5F) + \r  (95 in hex is 5F)
    print("\n[TEST] Injecting 95C Critical Temperature...")
    motor_bus.write(b't05015F\r')
    time.sleep(1.0)
    
    data = get_clean_packet(motor_bus)
    assert data is not None, "Failed to receive CAN frame from Arduino"
    assert data["MotorPWM"] == 0, f"Expected E-STOP (PWM 0), but got {data['MotorPWM']}"
    assert data["Temp"] == 95, f"Arduino did not acknowledge Temp override"