import subprocess
import random
import platform

# Configuration
num_clients = 10
num_attackers = 9

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
for client_id in range(1, num_clients + 1):
    is_attacker = "--attack" if client_id in attacker_ids else ""
    command = f"python client.py --id {client_id} {is_attacker}"

    if is_windows:
        command = f'start cmd /k {command}'

    print(f"Launching client {client_id} {'(attacker)' if is_attacker else '(benign)'}")
    subprocess.Popen(command, shell=True, executable="/bin/bash")
    print(f"Client {client_id} launched.")
    print("===================================")
