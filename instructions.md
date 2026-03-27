cd /home/syki/Desktop/canpinnmonitor
source venv/bin/activate
python3 pinnmotor.py

# 1. Apply the group change (System-wide)
sudo usermod -a -G dialout $USER

# 2. Refresh your group permissions (This might deactivate your venv)
newgrp dialout

# 3. Re-activate your venv (Crucial step!)
source venv/bin/activate

# 4. Now run your dashboard
python3 pinnmotor.py


test code:
sudo apt install python3-pytest
pytest -v -s pytest_hil.py 

pytest test_hil.py -v -s