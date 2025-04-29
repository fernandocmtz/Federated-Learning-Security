import subprocess
import random
import platform

# Configuration
num_clients = 10
num_attackers = 4

# Detect OS
current_os = platform.system().lower()
is_windows = current_os == "windows"

# Select attackers
attacker_ids = random.sample(range(1, num_clients + 1), num_attackers)
print(f"Attackers: {attacker_ids}")

# Save attacker IDs to file
with open("attacker_ids.txt", "w") as f:
    f.write(",".join(map(str, attacker_ids)))

# Launch each client

import platform, subprocess

is_windows = platform.system().lower() == "windows"

for client_id in range(1, num_clients + 1):
    attack_flag = "--attack" if client_id in attacker_ids else ""
    command     = f"python client.py --id {client_id} {attack_flag}"

    # --- spawn the process in a platform-friendly way ---
    if is_windows:
        # Opens a new Command Prompt window
        subprocess.Popen(f'start "" cmd /k {command}', shell=True)
    else:
        # Works on Linux/macOS; remove executable= if not needed
        subprocess.Popen(command, shell=True, executable="/bin/bash")

    role = "attacker" if attack_flag else "benign"
    print(f"Launching Client {client_id}  →  {role}")
    print("===================================")

