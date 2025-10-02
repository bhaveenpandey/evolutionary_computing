#2025PGCSIS18
#BHAVEEN PANDEY
#PARTICLE SWARM OPTIMIZATION
print("PARTICLE SWARM OPTIMIZATION")

import numpy as np
import random
import matplotlib.pyplot as plt

class ParticleSwarmOptimization:
    def __init__(self, seed=42):
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

        # PSO hyperparameters
        self.population_size = 30
        self.generations = 200
        self.w = 0.5      # inertia weight
        self.c1 = 1.5     # cognitive coefficient
        self.c2 = 1.5     # social coefficient

        np.random.seed(seed)
        random.seed(seed)
        self.validator_powers = np.random.uniform(50, 200, self.M)

        # Precompute normalizers for utility
        self.l_max = self.calculate_latency(self.M, self.X)
        self.eta_max = self.calculate_security(self.M)
        self.c_max = self.calculate_cost(self.v, self.t)

    # ---------------- Objective functions ----------------
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
        c_i = self.rho_i * x_i
        return float(np.sum(c_i) / n)

    def calculate_utility(self, m, n):
        L = self.calculate_latency(m, n)
        eta = self.calculate_security(m)
        C = self.calculate_cost(m, n)
        U = (0.33 * (L / self.l_max)
             + 0.33 * (self.eta_max / eta)
             + 0.34 * (C / self.c_max))
        return U, L, eta, C

    # ---------------- PSO ----------------
    def run(self, verbose=True, plot=True):
        # Initialize particles
        particles = [np.array([random.uniform(self.v, self.M),
                               random.uniform(self.t, self.X)])
                     for _ in range(self.population_size)]
        velocities = [np.array([random.uniform(-1, 1),
                                random.uniform(-10, 10)])
                      for _ in range(self.population_size)]
        pbest = [p.copy() for p in particles]
        pbest_scores = [self.calculate_utility(*p)[0] for p in particles]

        gbest_idx = int(np.argmin(pbest_scores))
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

        history_gbest = []
        history_avg_pbest = []
        history_current_best = []

        print("="*100)
        print(f"{'Gen':>4} | {'GlobalBest(m,n)':>20} | {'GlobalBest U':>12} | {'Avg PBest U':>12} | {'CurrentBest U':>12}")
        print("-"*100)

        for gen in range(1, self.generations+1):
            current_best_score = float('inf')
            current_best_particle = None

            for i in range(self.population_size):
                r1, r2 = random.random(), random.random()
                velocities[i] = (self.w * velocities[i]
                                 + self.c1 * r1 * (pbest[i] - particles[i])
                                 + self.c2 * r2 * (gbest - particles[i]))
                particles[i] = particles[i] + velocities[i]
                # Keep within bounds
                particles[i][0] = np.clip(particles[i][0], self.v, self.M)
                particles[i][1] = np.clip(particles[i][1], self.t, self.X)

                U, _, _, _ = self.calculate_utility(*particles[i])
                # Update personal best
                if U < pbest_scores[i]:
                    pbest[i] = particles[i].copy()
                    pbest_scores[i] = U
                # Update global best
                if U < gbest_score:
                    gbest = particles[i].copy()
                    gbest_score = U

                # Track current best in this generation
                if U < current_best_score:
                    current_best_score = U
                    current_best_particle = particles[i].copy()

            avg_pbest_score = np.mean(pbest_scores)
            history_gbest.append(gbest_score)
            history_avg_pbest.append(avg_pbest_score)
            history_current_best.append(current_best_score)

            if verbose:
                print(f"{gen:>4} | ({gbest[0]:.2f},{gbest[1]:.2f}) | {gbest_score:>12.6f} | {avg_pbest_score:>12.6f} | {current_best_score:>12.6f}")

        # Final best
        m_opt, n_opt = gbest
        U_opt, L_opt, eta_opt, C_opt = self.calculate_utility(m_opt, n_opt)

        print("-"*100)
        print("=== OPTIMIZATION COMPLETED ===")
        print(f"Best solution: m={m_opt:.2f}, n={n_opt:.2f}")
        print(f"Utility={U_opt:.6f}, Latency={L_opt:.6f}, η={eta_opt:.2f}, Cost={C_opt:.6f}")

        if plot:
            plt.figure(figsize=(10,6))
            plt.plot(history_gbest, label="Global Best Utility", color='blue')
            plt.plot(history_avg_pbest, label="Average Personal Best Utility", color='green')
            plt.plot(history_current_best, label="Current Best Utility", color='red')
            plt.xlabel("Generation")
            plt.ylabel("Utility (U)")
            plt.title("PSO Convergence")
            plt.grid(True)
            plt.legend()
            plt.show()

        return gbest, (m_opt, n_opt, U_opt, L_opt, eta_opt, C_opt), history_gbest

def main():
    pso = ParticleSwarmOptimization(seed=42)
    best_particle, details, history_gbest = pso.run(verbose=True, plot=True)
    return best_particle, details, history_gbest

if __name__ == "__main__":
    main()
