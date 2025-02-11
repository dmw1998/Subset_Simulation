import re
import numpy as np

def rRMSE(failure_probability):
    # input:
    # failure_probability
    
    # output:
    # relative rooted mean squared error
    
    failure_probability = np.array(failure_probability)
    
    difference = failure_probability - 1.6e-04
    
    expaction = np.mean(difference ** 2)
    
    return np.sqrt(expaction) / 1.6e-04

# Read data from the file
file_path = '/home/dmw/thesis_codes/Subset_Simulation/1d_example/subset_simulation.txt'
with open(file_path, 'r') as file:
    data = file.read()
    
matches = re.findall(r'\d+:\s*([\d\.eE+-]+)', data)
failure_probabilities = [float(match) for match in matches]

# # Use regex to find "failure probability" followed by a number in scientific notation
# matches = re.findall(r'failure probability ([\d\.eE+-]+)', data)

# # Convert matches to float for numerical operations, if needed
# failure_probabilities = [float(match) for match in matches]

for match in matches:
    match = float(match)
    
    if match < 9e-05 or match > 3e-04:
        failure_probabilities.remove(match)

print(len(failure_probabilities))
err = rRMSE(failure_probabilities)
print(err)