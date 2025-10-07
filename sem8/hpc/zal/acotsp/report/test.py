import subprocess
import csv
import random
import time

seeds = [random.randint(0, 100000) for _ in range(5)]
files = ["d198.tsp", "a280.tsp", "lin318.tsp", "pcb442.tsp", "rat783.tsp", "pr1002.tsp"]
types = ["WORKER", "QUEEN"]
alpha = 1
beta = 2
rho = 0.5

for file in files:
    for type in types:
        for seed in seeds:
            output_file = f"./out/output_{file}_{type}_{seed}.txt"
            input_file = file
            command = f"./acotsp ./in/{input_file} {output_file} {type} 1000 {alpha} {beta} {rho} {seed}"
            print(command)
            subprocess.run(command, shell=True)
            time.sleep(1)
            with open(output_file, "r") as f:
                result = f.readline().strip()
                with open("results.csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([file, seed, type, result])
