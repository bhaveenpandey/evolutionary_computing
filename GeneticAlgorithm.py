#2025PGCSIS18
#BHAVEEN PANDEY
#GENETIC_ALGORITHM 


import numpy as np
import random
import matplotlib.pyplot as plt

class BinaryCodedGA:
    def __init__(self, seed=42):
        # Problem parameters
        self.q = 4
        self.O = 0.5  #Mb
        self.theta = 1
        self.r_d = 1.2 #Mb/s
        self.r_u = 1.3 #Mb/s
        self.K = 100
        self.B = 0.5 #kb
        self.v = 2
        self.M = 10
        self.t = 1
        self.X = 500
        self.rho_i = 0.5
        self.psi = 0.001

        # GA hyperparameters
        self.m_min, self.m_max = self.v, self.M
        self.n_min, self.n_max = self.t, self.X
        self.alpha, self.beta, self.gamma = 0.33, 0.33, 0.34
        self.m_bits = 4
        self.n_bits = 9
        self.chromosome_length = self.m_bits + self.n_bits

        self.population_size = 100
        self.generations = 200
        self.crossover_rate = 0.9
        self.mutation_rate = 0.05
        self.tournament_size = 3
        self.elite_size = 5

        np.random.seed(seed)
        random.seed(seed)
        self.validator_powers = np.random.uniform(50, 200, self.M)

        # Precompute normalizers
        self.l_max = self.calculate_latency(self.M, self.X)
        self.eta_max = self.calculate_security(self.M)
        self.c_max = self.calculate_cost(self.v, self.t)

    # GA helpers
    def generate_random_chromosome(self):
        return ''.join(random.choice('01') for _ in range(self.chromosome_length))

    @staticmethod
    def binary_to_int(binary_str):
        return int(binary_str, 2) if binary_str else 0

    def decode_chromosome(self, chromosome):
        m_val = self.binary_to_int(chromosome[:self.m_bits])
        n_val = self.binary_to_int(chromosome[self.m_bits:])
        m = self.m_min + (m_val * (self.m_max - self.m_min)) // (2**self.m_bits - 1)
        n = self.n_min + (n_val * (self.n_max - self.n_min)) // (2**self.n_bits - 1)
        return int(max(self.m_min, min(self.m_max, m))), int(max(self.n_min, min(self.n_max, n)))

    # Objective functions
    def calculate_latency(self, m, n):
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

    # GA operators
    def fitness(self, chromosome):
        m, n = self.decode_chromosome(chromosome)
        if not (self.v <= m <= self.M and self.t <= n <= self.X):
            return 1e-6
        U, _, _, _ = self.calculate_utility(m, n)
        return 1.0 / (1.0 + U)

    def tournament_selection(self, population, fitness_values):
        idxs = random.sample(range(len(population)), self.tournament_size)
        return population[max(idxs, key=lambda i: fitness_values[i])]

    def uniform_crossover(self, p1, p2):
        if random.random() > self.crossover_rate:
            return p1, p2
        c1, c2 = [], []
        for a, b in zip(p1, p2):
            if random.random() < 0.5:
                c1.append(a); c2.append(b)
            else:
                c1.append(b); c2.append(a)
        return ''.join(c1), ''.join(c2)

    def mutate(self, chrom):
        return ''.join('1' if (bit == '0' and random.random() < self.mutation_rate) else
                       '0' if (bit == '1' and random.random() < self.mutation_rate) else
                       bit for bit in chrom)

    # Run GA
    def run(self, verbose=True, plot=True):
        population = [self.generate_random_chromosome() for _ in range(self.population_size)]
        best_chromosome, best_fitness = None, -1.0
        history = []

        print("="*90)
        print(f"{'Gen':>4} | {'m':>3} | {'n':>4} | {'Latency':>10} | {'Security':>10} | {'Cost':>10} | {'Utility':>12}")
        print("-"*90)

        for gen in range(1, self.generations + 1):
            fitness_values = [self.fitness(ch) for ch in population]
            best_idx = int(np.argmax(fitness_values))

            if fitness_values[best_idx] > best_fitness:
                best_fitness = fitness_values[best_idx]
                best_chromosome = population[best_idx]

            m_g, n_g = self.decode_chromosome(population[best_idx])
            U_g, L_g, eta_g, C_g = self.calculate_utility(m_g, n_g)
            history.append(U_g)

            if verbose:
                print(f"{gen:>4} | {m_g:>3} | {n_g:>4} | {L_g:>10.4f} | {eta_g:>10.2f} | {C_g:>10.4f} | {U_g:>12.6f}")

            # elitism
            elite_idxs = np.argsort(fitness_values)[-self.elite_size:]
            elites = [population[i] for i in elite_idxs]
            new_pop = elites[:]

            while len(new_pop) < self.population_size:
                p1 = self.tournament_selection(population, fitness_values)
                p2 = self.tournament_selection(population, fitness_values)
                c1, c2 = self.uniform_crossover(p1, p2)
                new_pop.append(self.mutate(c1))
                if len(new_pop) < self.population_size:
                    new_pop.append(self.mutate(c2))

            population = new_pop

        # final best
        m_opt, n_opt = self.decode_chromosome(best_chromosome)
        U_opt, L_opt, eta_opt, C_opt = self.calculate_utility(m_opt, n_opt)

        print("-"*90)
        print("=== OPTIMIZATION COMPLETED ===")
        print(f"Best solution: m={m_opt}, n={n_opt}")
        print(f"Utility={U_opt:.6f}, Latency={L_opt:.6f}, η={eta_opt:.2f}, Cost={C_opt:.6f}")

        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(history, label="Best Utility per Generation", color='blue')
            plt.xlabel("Generation")
            plt.ylabel("Utility (U)")
            plt.title("GA Convergence (Utility vs Generation)")
            plt.grid(True)
            plt.legend()
            plt.show()

        return best_chromosome, (m_opt, n_opt, U_opt, L_opt, eta_opt, C_opt), history


def main():
    ga = BinaryCodedGA(seed=42)
    best_chromosome, details, history = ga.run(verbose=True, plot=True)
    return best_chromosome, details, history

if __name__ == "__main__":
    main()
