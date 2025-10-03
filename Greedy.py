#2025PGCSIS18
#BHAVEEN PANDEY
#GENETIC_ALGORITHM 


import numpy as np
import matplotlib.pyplot as plt

print("Greedy Method")

class BlockchainGreedy:
    def __init__(self, seed=42):
        # Simulation parameters
        self.q = 4
        self.O = 0.5       # Mb
        self.theta = 1
        self.r_d = 1.2     # Mb/s
        self.r_u = 1.3     # Mb/s
        self.K = 100
        self.B = 0.5       # Mb
        self.v = 2
        self.M = 10
        self.t = 1
        self.X = 500
        self.rho_i = 0.5
        self.psi = 0.001

        # Weight coefficients
        self.alpha = 0.33
        self.beta = 0.33
        self.gamma = 0.34

        # Random seed and validator powers
        np.random.seed(seed)
        self.validator_powers = np.random.uniform(50, 200, self.M)

        # Precompute normalizers
        self.l_max = self.calculate_latency(self.M, self.X)
        self.eta_max = self.calculate_security(self.M)
        self.c_max = self.calculate_cost(self.v, self.t)

    # ---------------- Objective functions ----------------
    def calculate_latency(self, m, n):
        x_i = self.validator_powers[:m]
        download_time = (n * self.B) / self.r_d
        processing_time = np.max(self.K / x_i)
        overhead = self.psi * (n * self.B) * m
        upload_time = self.O / self.r_u
        return download_time + processing_time + overhead + upload_time

    def calculate_security(self, m):
        return self.theta * (m ** self.q)

    def calculate_cost(self, m, n):
        x_i = self.validator_powers[:m]
        c_i = self.rho_i * x_i
        return np.sum(c_i) / n

    def calculate_utility(self, m, n):
        L = self.calculate_latency(m, n)
        eta = self.calculate_security(m)
        C = self.calculate_cost(m, n)
        U = (self.alpha * (L / self.l_max)
             + self.beta * (self.eta_max / eta)
             + self.gamma * (C / self.c_max))
        return U, L, eta, C

    # ---------------- Greedy search ----------------
    def greedy_search(self):
        best_U = float('inf')
        best_m, best_n = None, None
        history = []

        iteration = 0
        print("="*70)
        print(f"{'Iteration':>9} | {'m':>2} | {'n':>3} | {'Utility':>10}")
        print("-"*70)

        for m in range(self.v, self.M + 1):
            for n in range(self.t, self.X + 1, 5):  # step=5 → ~900 iterations
                iteration += 1
                U, L, eta, C = self.calculate_utility(m, n)
                history.append(U)
                if U < best_U:
                    best_U = U
                    best_m, best_n = m, n

                # Print every iteration
                print(f"{iteration:9d} | {m:2d} | {n:3d} | {U:10.6f}")

        print("\n=== Greedy Optimization Completed ===")
        print(f"Total iterations: {iteration}")
        print(f"Best solution: m={best_m}, n={best_n}")
        L_opt, eta_opt, C_opt = self.calculate_latency(best_m, best_n), self.calculate_security(best_m), self.calculate_cost(best_m, best_n)
        print(f"Utility={best_U:.6f}, Latency={L_opt:.6f}, η={eta_opt:.2f}, Cost={C_opt:.6f}")

        # Plot convergence
        plt.figure(figsize=(8,5))
        plt.plot(history, label="Utility over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Utility (U)")
        plt.title("Greedy Heuristic Utility Convergence")
        plt.grid(True)
        plt.legend()
        plt.show()

        return best_m, best_n, best_U, history

# ---------------- Run the greedy optimizer ----------------
if __name__ == "__main__":
    optimizer = BlockchainGreedy(seed=42)
    best_m, best_n, best_U, history = optimizer.greedy_search()
