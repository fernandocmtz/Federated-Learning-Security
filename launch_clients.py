import subprocess
import random

# Configuration
num_clients = 10
num_attackers = 9

# Pick attacker client IDs (can be random or predefined)
attacker_ids = random.sample(range(1, num_clients + 1), num_attackers)

# Print the attacker IDs for reference
print(f"Attackers: {attacker_ids}")

# Run each client one at a time
for client_id in range(1, num_clients + 1):
        

    is_attacker = "--attack" if client_id in attacker_ids else ""
    
    # Create the command string
    command = f'start cmd /k python client.py --id {client_id} {is_attacker}'
    
    print(f"Launching client {client_id} {'(attacker)' if is_attacker else '(benign)'}")

    # Run it in a new terminal window
    subprocess.Popen(command, shell=True)


    print(f"Client {client_id} finished.")
    print("===================================")
    