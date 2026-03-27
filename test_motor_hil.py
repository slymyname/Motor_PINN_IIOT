import serial
import time
import pytest

PORT = '/dev/ttyS0' # Ensure this matches your Proteus virtual port
BAUD = 115200

@pytest.fixture(scope="module")
def hardware():
    print(f"\n[SETUP] Connecting to Digital Twin on {PORT}...")
    ser = serial.Serial(PORT, BAUD, timeout=2)
    
    # --- THE FIX: COLD BOOT DELAY ---
    # Opening a serial port resets the microcontroller. 
    # We must give FreeRTOS 3.5 seconds to fully boot the scheduler!
    time.sleep(3.5) 
    
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    
    yield ser 
    
    print("\n[TEARDOWN] Disconnecting from Digital Twin...")
    ser.write(b't0510\r') 
    ser.close()

# =====================================================================
# HIL AUTOMATION HELPERS
# =====================================================================

def send_can_override(ser, temp_val):
    """Formats and injects an SLCAN message to simulate sensor data."""
    if temp_val is None:
        ser.write(b't0510\r') 
    else:
        hex_temp = f"{temp_val:02X}"
        command = f"t0501{hex_temp}\r".encode('utf-8')
        ser.write(command)

def verify_sustained_state(ser, exp_temp, exp_pwm, exp_status, duration_sec):
    """
    1. Settling Phase: Waits for the queue to drain and the target to be reached.
    2. Monitoring Phase: Strictly ensures the state is sustained without flickering.
    """
    # --- 1. SETTLING PHASE (Increased to 5.0s for slow simulators) ---
    settling_timeout = time.time() + 5.0
    state_reached = False
    last_temp = None
    
    while time.time() < settling_timeout:
        if ser.in_waiting > 0:
            line = ser.read_until(b'\r').decode('utf-8', errors='ignore').strip()
            if line.startswith('t1004') and len(line) >= 13:
                last_temp = int(line[5:7], 16)
                if last_temp == exp_temp:
                    state_reached = True
                    break # Target reached! Exit settling phase.
                    
    assert state_reached, f"System failed to settle! Expected {exp_temp}C, stuck at {last_temp}C."
    
    # --- 2. SUSTAINED MONITORING PHASE ---
    monitor_end = time.time() + duration_sec
    frames_checked = 1 
    
    while time.time() < monitor_end:
        if ser.in_waiting > 0:
            line = ser.read_until(b'\r').decode('utf-8', errors='ignore').strip()
            
            if line.startswith('t1004') and len(line) >= 13:
                temp = int(line[5:7], 16)
                pwm = int(line[9:11], 16)
                status = int(line[11:13], 16)
                
                # Assertions will immediately fail the test if the RTOS flickers
                assert temp == exp_temp, f"Temperature fluctuated! Expected {exp_temp}, got {temp}"
                assert status == exp_status, f"Status fluctuated! Expected {exp_status}, got {status}"
                assert pwm == exp_pwm, f"PWM fluctuated! Expected {exp_pwm}, got {pwm}"
                
                frames_checked += 1

    assert frames_checked >= 4, f"Error: Only received {frames_checked} frames. RTOS might be frozen!"
    print(f" -> Successfully verified {frames_checked} sequential frames over {duration_sec}s.")

# =====================================================================
# THE TEST SUITE
# =====================================================================

def test_1_baseline_stability(hardware):
    print("\n--- Running Test 1: Baseline Stability (5 Seconds) ---")
    send_can_override(hardware, 25) 
    verify_sustained_state(hardware, exp_temp=25, exp_pwm=255, exp_status=0, duration_sec=5.0)

def test_2_thermal_throttling(hardware):
    print("\n--- Running Test 2: Thermal Throttling / Warning State (5 Seconds) ---")
    send_can_override(hardware, 75) 
    verify_sustained_state(hardware, exp_temp=75, exp_pwm=128, exp_status=1, duration_sec=5.0)

def test_3_critical_thermal_estop(hardware):
    print("\n--- Running Test 3: Critical E-STOP (5 Seconds) ---")
    send_can_override(hardware, 105) 
    verify_sustained_state(hardware, exp_temp=105, exp_pwm=0, exp_status=2, duration_sec=5.0)

def test_4_can_flood_and_recovery(hardware):
    print("\n--- Running Test 4: CAN Bus Jitter & RTOS Memory Flood (4 Seconds) ---")
    
    # 1. Spam the queue to completely fill the 64-byte UART buffer and 5-item RTOS queue
    for _ in range(10):
        hardware.write(b't050119\r') # 25C
        hardware.write(b't050164\r') # 100C
        
    print(" -> Flood complete. Waiting for Arduino hardware buffers to drain...")
    time.sleep(3.0) # Let the Arduino process whatever survived the buffer overflow
    
    # Throw away all the chaotic telemetry generated during the flood
    hardware.reset_input_buffer() 
    
    # 2. NOW command it to stabilize at 25C (after the bus is clear!)
    print(" -> Bus is clear. Sending final recovery command...")
    send_can_override(hardware, 25)
    
    # 3. Verify it safely settles and sustains
    verify_sustained_state(hardware, exp_temp=25, exp_pwm=255, exp_status=0, duration_sec=4.0)