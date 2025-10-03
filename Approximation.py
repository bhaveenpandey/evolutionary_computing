# 2025PGCSIS18
# BHAVEEN PANDEY
# APPROXIMATION ALGORITHM
print("APPROXIMATION ALGORITHM (GRID SEARCH)")

import numpy as np
import matplotlib.pyplot as plt

class ApproximationOptimization:
    def __init__(self):
        # Problem parameters
        self.q = 4
        self.O = 0.5  # Mb
        self.theta = 1
        self.r_d = 1.2  # Mb/s
        self.r_u = 1.3  # Mb/s
        self.K = 100
        self.B = 0.5  # kb
        self.v = 2
        self.M = 10
        self.t = 1
        self.X = 500
        self.rho_i = 0.5
        self.psi = 0.001

        # Approximation parameters
        self.step_m = 1      # step for validators
        self.step_n = 10     # step for transactions per block
        self.validator_powers = np.random.uniform(50, 200, self.M)

        # Precompute normalizers
        self.l_max = self.calculate_latency(self.M, self.X)
        self.eta_max = self.calculate_security(self.M)
        self.c_max = self.calculate_cost(self.v, self.t)

    # --- Objective functions ---
    def calculate_latency(self, m, n):
        m = max(1, int(round(m)))
        x_i = self.validator_powers[:m]
        download_time = (n * self.B) / self.r_d
        processing_time = float(np.max(self.K / x_i))
        overhead = self.psi * (n * self.B) * m
        upload_time = self.O / self.r_u
        return download_time + processing_time + overhead + upload_time

    def calculate_security(self, m):
        return self.theta * (m ** self.q)

    def calculate_cost(self, m, n):
        m = max(1, int(round(m)))
        x_i = self.validator_powers[:m]
        return float(np.sum(self.rho_i * x_i) / n)

    def calculate_utility(self, m, n):
        L = self.calculate_latency(m, n)
        eta = self.calculate_security(m)
        C = self.calculate_cost(m, n)
        U = (0.33 * (L / self.l_max)
             + 0.33 * (self.eta_max / eta)
             + 0.34 * (C / self.c_max))
        return U, L, eta, C

    # --- Approximation / Grid Search ---
    def run(self, verbose=True, plot=True):
        best_m, best_n = self.v, self.t
        best_U, best_L, best_eta, best_C = float('inf'), None, None, None
        history_U = []

        m_values = np.arange(self.v, self.M+1, self.step_m)
        n_values = np.arange(self.t, self.X+1, self.step_n)

        iteration = 0
        print("="*100)
        print(f"{'Iter':>4} | {'m':>3} | {'n':>4} | {'Latency':>10} | {'Security':>10} | {'Cost':>10} | {'Utility':>12}")
        print("-"*100)

        for m in m_values:
            for n in n_values:
                iteration += 1
                U, L, eta, C = self.calculate_utility(m, n)
                history_U.append(U)
                if U < best_U:
                    best_U = U
                    best_m, best_n = m, n
                    best_L, best_eta, best_C = L, eta, C
                if verbose:
                    print(f"{iteration:>4} | {m:>3} | {n:>4} | {L:>10.4f} | {eta:>10.2f} | {C:>10.4f} | {U:>12.6f}")

        print("-"*100)
        print("=== OPTIMIZATION COMPLETED ===")
        print(f"Best solution: m={best_m}, n={best_n}")
        print(f"Utility={best_U:.6f}, Latency={best_L:.6f}, η={best_eta:.2f}, Cost={best_C:.6f}")

        if plot:
            plt.figure(figsize=(10,6))
            plt.plot(history_U, label="Utility per grid point", color='blue')
            plt.xlabel("Iteration (grid points)")
            plt.ylabel("Utility (U)")
            plt.title("Approximation Algorithm Convergence")
            plt.grid(True)
            plt.legend()
            plt.show()

        return (best_m, best_n, best_U, best_L, best_eta, best_C), history_U

def main():
    approx = ApproximationOptimization()
    best_solution, history_U = approx.run(verbose=True, plot=True)
    return best_solution, history_U

if __name__ == "__main__":
    main()
