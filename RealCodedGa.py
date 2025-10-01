#2025PGCSIS18
#BHAVEEN PANDEY
#REAL-VALUED GENETIC_ALGORITHM

print("REAL CODED GENETIC ALGORITHM")

import numpy as np
import random
import matplotlib.pyplot as plt

class RealCodedGA:
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

        # GA hyperparameters
        self.alpha, self.beta, self.gamma = 0.33, 0.33, 0.34
        self.population_size = 100
        self.generations = 200
        self.crossover_rate = 0.9
        self.mutation_rate = 0.1
        self.tournament_size = 3
        self.elite_size = 5

        np.random.seed(seed)
        random.seed(seed)
        self.validator_powers = np.random.uniform(50, 200, self.M)

        # Precompute normalizers
        self.l_max = self.calculate_latency(self.M, self.X)
        self.eta_max = self.calculate_security(self.M)
        self.c_max = self.calculate_cost(self.v, self.t)

    # ---------------- Objective functions ----------------
    def calculate_latency(self, m, n):
        m = max(1, int(round(m)))
        if m <= 0:
            return float('inf')
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
        if n <= 0:
            return float('inf')
        x_i = self.validator_powers[:m]
        c_i = self.rho_i * x_i
        return float(np.sum(c_i) / n)

    def calculate_utility(self, m, n):
        L = self.calculate_latency(m, n)
        eta = self.calculate_security(m)
        C = self.calculate_cost(m, n)
        U = (self.alpha * (L / self.l_max)
             + self.beta * (self.eta_max / eta)
             + self.gamma * (C / self.c_max))
        return U, L, eta, C

    # ---------------- GA operators ----------------
    def fitness(self, individual):
        m, n = individual
        if not (self.v <= m <= self.M and self.t <= n <= self.X):
            return 1e-6
        U, _, _, _ = self.calculate_utility(m, n)
        return 1.0 / (1.0 + U)

    def tournament_selection(self, population, fitness_values):
        idxs = random.sample(range(len(population)), self.tournament_size)
        return population[max(idxs, key=lambda i: fitness_values[i])].copy()

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            mutation = np.random.normal(0, 1, size=2)  # small perturbation
            individual = individual + mutation
            individual[0] = np.clip(individual[0], self.v, self.M)
            individual[1] = np.clip(individual[1], self.t, self.X)
        return individual

    # ---------------- Run GA ----------------
    def run(self, verbose=True, plot=True):
        # Initialize population with real values [m, n]
        population = [np.array([random.uniform(self.v, self.M),
                                random.uniform(self.t, self.X)]) 
                      for _ in range(self.population_size)]

        best_chromosome = None
        best_fitness = -1
        history = []

        print("="*90)
        print(f"{'Gen':>4} | {'m':>6} | {'n':>6} | {'Latency':>10} | {'Security':>10} | {'Cost':>10} | {'Utility':>12}")
        print("-"*90)

        for gen in range(1, self.generations+1):
            fitness_values = [self.fitness(ind) for ind in population]
            best_idx = int(np.argmax(fitness_values))

            if fitness_values[best_idx] > best_fitness:
                best_fitness = fitness_values[best_idx]
                best_chromosome = population[best_idx].copy()

            m_g, n_g = population[best_idx]
            U_g, L_g, eta_g, C_g = self.calculate_utility(m_g, n_g)
            history.append(U_g)

            if verbose:
                print(f"{gen:>4} | {m_g:>6.2f} | {n_g:>6.2f} | {L_g:>10.4f} | {eta_g:>10.2f} | {C_g:>10.4f} | {U_g:>12.6f}")

            # Elitism
            elite_idxs = np.argsort(fitness_values)[-self.elite_size:]
            elites = [population[i].copy() for i in elite_idxs]
            new_pop = elites[:]

            while len(new_pop) < self.population_size:
                p1 = self.tournament_selection(population, fitness_values)
                p2 = self.tournament_selection(population, fitness_values)
                c1, c2 = self.crossover(p1, p2)
                new_pop.append(self.mutate(c1))
                if len(new_pop) < self.population_size:
                    new_pop.append(self.mutate(c2))

            population = new_pop

        # Final best
        m_opt, n_opt = best_chromosome
        U_opt, L_opt, eta_opt, C_opt = self.calculate_utility(m_opt, n_opt)

        print("-"*90)
        print("=== OPTIMIZATION COMPLETED ===")
        print(f"Best solution: m={m_opt:.2f}, n={n_opt:.2f}")
        print(f"Utility={U_opt:.6f}, Latency={L_opt:.6f}, η={eta_opt:.2f}, Cost={C_opt:.6f}")

        if plot:
            plt.figure(figsize=(8,5))
            plt.plot(history, label="Best Utility per Generation", color='blue')
            plt.xlabel("Generation")
            plt.ylabel("Utility (U)")
            plt.title("Real-Coded GA Convergence")
            plt.grid(True)
            plt.legend()
            plt.show()

        return best_chromosome, (m_opt, n_opt, U_opt, L_opt, eta_opt, C_opt), history


def main():
    ga = RealCodedGA(seed=42)
    best_chromosome, details, history = ga.run(verbose=True, plot=True)
    return best_chromosome, details, history


if __name__ == "__main__":
    main()
