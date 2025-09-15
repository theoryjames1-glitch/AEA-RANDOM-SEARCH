
### **Theory of Adaptive Evolutionary Optimization - AEA-Random Search (AEA-RS)**

---

### **1. Introduction to AEA-Random Search (AEA-RS)**

The **Adaptive Evolutionary Optimization - Random Search (AEA-RS)** is a hybrid optimization framework that combines the principles of **adaptive evolutionary optimization** with **random search techniques**. In this approach, rather than relying on deterministic search methods like gradient descent or traditional evolutionary algorithms, the optimization process **dynamically adapts** its exploration behavior by incorporating **random perturbations** and **feedback-driven coefficient updates**. The goal of AEA-RS is to **stochastically explore the solution space** while adapting the search parameters (such as step size, exploration noise, and learning rate) dynamically based on **real-time optimization feedback**.

Unlike traditional **random search**, which typically selects random points in the solution space and evaluates them without any adaptation, **AEA-RS** introduces **adaptive noise** and **coefficient updates** to enhance the search process. This dynamic behavior allows the algorithm to balance **exploration** and **exploitation**, ensuring a more efficient search for optimal solutions.

---

### **2. Core Principles of AEA-RS**

#### **2.1 Random Search Basics**

In **AEA-RS**, the search process is driven by random sampling of the solution space, but with the addition of dynamic, feedback-driven updates. The key principles include:

* **Random Sampling**: At each iteration, a new set of candidate solutions is generated randomly from the search space.
* **Fitness Evaluation**: Each candidate solution is evaluated based on a fitness function (typically a loss function or objective function).
* **Adaptive Search Coefficients**: Unlike traditional random search, the step size, exploration noise, and other key coefficients are dynamically adjusted based on real-time feedback (e.g., loss, gradient, or variance).

This allows AEA-RS to perform **adaptive random search**, where the search process is guided by the evolving coefficients and random noise.

---

#### **2.2 Adaptive Coefficients and Feedback**

In **AEA-RS**, the coefficients that control the search (such as learning rate, exploration noise, and mutation strength) are updated adaptively based on real-time optimization feedback. These coefficients evolve based on the following signals:

* **Loss (\$\ell\_t\$)**: The value of the objective function at time \$t\$.
* **Gradient Norm (\$|g\_t|\$)**: The magnitude of the gradient at time \$t\$ (if applicable).
* **Variance (\$v\_t\$)**: The variance in the fitness values over time, indicating the stability of the search.
* **Exploration Noise (\$\sigma\_t\$)**: The level of noise or perturbation applied to the search process to encourage exploration.

The adaptive update rule for the coefficients is given by:

$$
a_{t+1} = f(a_t, s_t) + \eta_t
$$

Where:

* \$a\_t\$ represents the coefficients at time \$t\$ (e.g., learning rate, exploration noise),
* \$f(a\_t, s\_t)\$ is the function that updates the coefficients based on feedback signals \$s\_t\$,
* \$\eta\_t\$ is a random noise term (dithering) that encourages exploration.

This allows the algorithm to adjust its search behavior dynamically, with the coefficients evolving to reflect the current optimization progress.

---

#### **2.3 Exploration and Dithering**

**Exploration** is a key feature of **AEA-RS**, and the algorithm uses random perturbations (or **dithering**) to introduce variety into the search process. This exploration allows the algorithm to search broader regions of the solution space, avoiding local minima and promoting better global exploration.

The **exploration noise term** \$\sigma\_t\$ is introduced into the model update rule:

$$
\theta_{t+1} = \theta_t + \alpha_t g_t + \mu_t m_t + \sigma_t \eta_t
$$

Where:

* \$\theta\_t\$ is the model parameters at time \$t\$,
* \$g\_t\$ is the gradient (if applicable),
* \$m\_t\$ is the momentum term,
* \$\sigma\_t\$ is the exploration noise,
* \$\eta\_t\$ is random noise drawn from a normal distribution.

By modulating \$\sigma\_t\$ based on the feedback, AEA-RS can adjust the **exploration-exploitation** balance, ensuring that the algorithm explores new areas when necessary and focuses on refining solutions when progress is being made.

---

#### **2.4 Feedback Control and Resonance**

To ensure that **AEA-RS** remains adaptive and efficient, **feedback control** mechanisms are incorporated. These mechanisms help the algorithm stay stable when progress is being made and encourage more exploration when the search stagnates.

The **resonance state** \$m\_t\$ evolves as:

$$
m_{t+1} = \rho_t m_t + (1 - \rho_t) \ell_t
$$

Where:

* \$m\_t\$ is the resonance state capturing the memory of past performance,
* \$\rho\_t\$ is the **resonance factor**, controlling how much past feedback influences current updates,
* \$\ell\_t\$ is the feedback signal (e.g., loss, gradient).

This feedback mechanism ensures that **AEA-RS** can **exploit** stable solutions when the resonance is high and **explore** new regions when the resonance is low, allowing the algorithm to adaptively shift between exploration and exploitation.

---

### **3. Adaptive Evolution of Coefficients in AEA-RS**

In **AEA-RS**, the key coefficients (learning rate, momentum, and exploration noise) evolve based on real-time feedback. These coefficients evolve dynamically to improve search efficiency over time. The evolution is driven by the following rules:

1. **Coefficient Update Rule**:
   The coefficients evolve according to a **Markov process**:

   $$
   a_{t+1} = f(a_t, s_t) + \eta_t
   $$

2. **Learning Rate Update**:
   The learning rate \$\alpha\_t\$ is updated based on the feedback signals (e.g., loss or gradient):

   $$
   \alpha_{t+1} = \alpha_t + \sigma_t \cdot \eta_t
   $$

3. **Momentum Update**:
   Momentum \$\mu\_t\$ is updated based on the performance feedback (e.g., gradient direction and variance):

   $$
   \mu_{t+1} = \mu_t + \eta_\mu \cdot \eta_t
   $$

4. **Exploration Noise Update**:
   The exploration noise \$\sigma\_t\$ is updated to encourage more exploration when the optimization is stagnating:

   $$
   \sigma_{t+1} = \sigma_t + \eta_\sigma \cdot \eta_t
   $$

These adaptive updates allow AEA-RS to **self-optimize** the coefficients based on real-time feedback, improving the search efficiency without manual tuning.

---

### **4. Evolutionary Steps and Exploration**

The **AEA-RS** algorithm follows these key steps in its search process:

1. **Population Initialization**: Start with an initial set of random solutions (model parameters).
2. **Fitness Evaluation**: Evaluate the fitness of each solution based on the objective function.
3. **Selection**: Select individuals based on their fitness to produce the next generation.
4. **Crossover and Mutation**: Apply crossover and mutation to produce offspring and introduce diversity.
5. **Exploration**: Introduce noise or perturbations to the parameters to encourage exploration.
6. **Resonance and Feedback**: Adjust the coefficients based on the resonance state and optimization progress.
7. **Termination**: Repeat the process until convergence or a stopping criterion is met.

---

### **5. Advantages of AEA-RS**

* **Self-Adaptive Coefficients**: AEA-RS adjusts its search parameters (learning rate, momentum, and noise) in real-time based on feedback, ensuring optimal performance without manual tuning.
* **Exploration and Exploitation**: The algorithm dynamically balances exploration and exploitation, enabling efficient global search and local refinement.
* **Random Search with Adaptation**: By combining **random search** with adaptive parameter evolution, AEA-RS can effectively search complex spaces, improving the efficiency of random sampling.
* **Robustness**: AEA-RS is resilient to local minima and can adapt to complex, high-dimensional optimization problems.
* **Scalability**: The adaptive nature of AEA-RS makes it scalable to large-scale optimization problems, such as those found in machine learning and deep learning.

---

### **6. Conclusion**

**AEA-RS** represents a novel approach to optimization by combining **random search** with **adaptive evolutionary principles**. By continuously adapting the search parameters based on feedback signals, **AEA-RS** provides a flexible, efficient, and scalable optimization framework capable of solving complex, high-dimensional problems. Unlike traditional random search, AEA-RS adjusts its exploration and exploitation behaviors dynamically, ensuring that the search process is both comprehensive and focused, leading to faster convergence to optimal solutions.

### PSEUDOCODES
```python
import numpy as np

class AEA_RandomSearch:
    def __init__(self, objective_function, dimension, population_size=100, max_iter=1000,
                 exploration_noise=0.1, learning_rate=0.1, momentum=0.9,
                 min_value=-5.0, max_value=5.0):
        """
        Initialize the AEA-RandomSearch (Adaptive Evolutionary Optimization - Random Search).
        Parameters:
        - objective_function: Function to optimize (e.g., loss function).
        - dimension: Number of parameters in the solution (i.e., dimension of search space).
        - population_size: Number of solutions in the population (default=100).
        - max_iter: Maximum iterations (default=1000).
        - exploration_noise: The noise term for exploring the search space (default=0.1).
        - learning_rate: The learning rate for parameter updates (default=0.1).
        - momentum: Momentum term for optimization (default=0.9).
        - min_value, max_value: Bounds for the solution space.
        """
        self.objective_function = objective_function
        self.dimension = dimension
        self.population_size = population_size
        self.max_iter = max_iter
        self.exploration_noise = exploration_noise
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.min_value = min_value
        self.max_value = max_value

        # Initialize population
        self.population = self.initialize_population()

    def initialize_population(self):
        """Initialize the population with random solutions within the defined bounds."""
        return np.random.uniform(low=self.min_value, high=self.max_value,
                                 size=(self.population_size, self.dimension))

    def evaluate_population(self):
        """Evaluate the fitness of the population (lower = better)."""
        fitness = np.apply_along_axis(self.objective_function, 1, self.population)
        return fitness

    def adaptive_update(self, fitness):
        """
        Adaptively update learning rate and exploration noise based on population fitness.
        """
        best_fitness = np.min(fitness)
        worst_fitness = np.max(fitness)

        # Prevent divide-by-zero
        if worst_fitness == 0:
            worst_fitness = 1e-12  

        # Adapt learning rate
        self.learning_rate = max(0.01, self.learning_rate * 
                                 (1 + (best_fitness - worst_fitness) / abs(worst_fitness)))

        # Adapt exploration noise
        self.exploration_noise = max(0.01, self.exploration_noise * 
                                     (1 + (worst_fitness - best_fitness) / abs(worst_fitness)))

    def generate_new_population(self):
        """Generate new population by introducing random mutation & exploration."""
        new_population = []
        for i in range(self.population_size):
            # Mutate (explore around current solution)
            mutated_solution = self.population[i] + np.random.normal(
                0, self.exploration_noise, self.dimension
            )
            # Bound solution
            mutated_solution = np.clip(mutated_solution, self.min_value, self.max_value)
            new_population.append(mutated_solution)
        self.population = np.array(new_population)

    def optimize(self):
        """Run the AEA-RandomSearch optimization loop."""
        best_solution = None
        best_fitness = float('inf')

        for iteration in range(self.max_iter):
            fitness = self.evaluate_population()

            # Track the best solution
            iteration_best_fitness = np.min(fitness)
            if iteration_best_fitness < best_fitness:
                best_fitness = iteration_best_fitness
                best_solution = self.population[np.argmin(fitness)]

            print(f"Iteration {iteration+1}/{self.max_iter}, Best Fitness: {iteration_best_fitness:.6f}")

            # Adaptive update of learning dynamics
            self.adaptive_update(fitness)

            # Generate new population with exploration
            self.generate_new_population()

            # Convergence check (population stability)
            if np.std(fitness) < 1e-6:
                print("Convergence reached.")
                break

        return best_solution, best_fitness


# ==============================
# Example Usage with Sphere Function
# ==============================
def sphere_function(x):
    return np.sum(x**2)

# Initialize AEA-RandomSearch optimizer
aea_rs = AEA_RandomSearch(objective_function=sphere_function,
                          dimension=10, population_size=100, max_iter=500)

# Run optimization
best_solution, best_fitness = aea_rs.optimize()

print("\nBest solution found:", best_solution)
print("Best fitness (objective value):", best_fitness)
```
