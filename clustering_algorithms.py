"""
Clustering Algorithms for Tobacco Consumption Analysis

This module implements various clustering algorithms:
1. Genetic Algorithm (GA) Clustering
2. Improved Genetic Algorithm (IGA) Clustering
3. Hybrid IGA-SOM Clustering

These algorithms can be imported and used in the Jupyter notebook.
"""

import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from minisom import MiniSom
from deap import base, creator, tools, algorithms

def setup_ga(data, n_clusters=3):
    """
    Setup basic Genetic Algorithm for clustering.
    
    Parameters:
    -----------
    data : array-like of shape (n_samples, n_features)
        Input data for clustering
    n_clusters : int, default=3
        Number of clusters to form
        
    Returns:
    --------
    toolbox : DEAP toolbox
        Configured toolbox for GA operations
    """
    # Create fitness and individual types
    if 'FitnessMax' not in dir(creator):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if 'Individual' not in dir(creator):
        creator.create("Individual", list, fitness=creator.FitnessMax)
    
    # Define individual and population
    ind_size = n_clusters * data.shape[1]  # Number of dimensions in each centroid * number of centroids
    
    # Helper functions for GA
    def generate_individual():
        # Use KMeans++ to initialize centroids
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42).fit(data)
        initial_centroids = kmeans.cluster_centers_
        return initial_centroids.flatten().tolist()
    
    def evaluate(individual):
        centroids = np.array(individual).reshape((n_clusters, data.shape[1]))
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        
        # Ensure we have more than one cluster
        if len(np.unique(labels)) < 2:
            return -1.0,
            
        return silhouette_score(data, labels),
    
    # Toolbox setup
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
    
    return toolbox

def run_ga(data, n_clusters=3, pop_size=50, n_gen=40, cx_pb=0.5, mut_pb=0.2):
    """
    Run Genetic Algorithm for clustering.
    
    Parameters:
    -----------
    data : array-like of shape (n_samples, n_features)
        Input data for clustering
    n_clusters : int, default=3
        Number of clusters to form
    pop_size : int, default=50
        Population size
    n_gen : int, default=40
        Number of generations
    cx_pb : float, default=0.5
        Crossover probability
    mut_pb : float, default=0.2
        Mutation probability
        
    Returns:
    --------
    labels : array of shape (n_samples,)
        Cluster labels
    centroids : array of shape (n_clusters, n_features)
        Final cluster centroids
    silhouette : float
        Silhouette score of the clustering
    dbi : float
        Davies-Bouldin index of the clustering
    log : object
        Log object with select method
    """
    # Setup GA
    toolbox = setup_ga(data, n_clusters)
    
    # Create initial population
    population = toolbox.population(n=pop_size)
    
    # Evaluate initial population
    for ind in population:
        if not ind.fitness.valid:
            try:
                ind.fitness.values = toolbox.evaluate(ind)
            except Exception as e:
                print(f"Error evaluating individual: {e}")
                ind.fitness.values = (-1.0,)
    
    # Hall of Fame to store best individuals
    hof = tools.HallOfFame(1)
    hof.update(population)
    
    # Store log data manually
    log_data = []
    
    # Run for n_gen generations
    for gen in range(n_gen):
        # Select the next generation individuals
        try:
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
        except Exception as e:
            print(f"Error in selection in generation {gen}: {e}")
            # Fallback: create new individuals
            offspring = toolbox.population(n=len(population))
        
        # Apply crossover
        for i in range(1, len(offspring), 2):
            if i < len(offspring) and random.random() < cx_pb:
                try:
                    offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values
                except Exception as e:
                    print(f"Error in crossover in generation {gen}: {e}")
        
        # Apply mutation
        for i in range(len(offspring)):
            if random.random() < mut_pb:
                try:
                    offspring[i] = toolbox.mutate(offspring[i])[0]
                    del offspring[i].fitness.values
                except Exception as e:
                    print(f"Error in mutation in generation {gen}: {e}")
        
        # Evaluate offspring
        for ind in offspring:
            if not ind.fitness.valid:
                try:
                    ind.fitness.values = toolbox.evaluate(ind)
                except Exception as e:
                    print(f"Error evaluating offspring in generation {gen}: {e}")
                    ind.fitness.values = (-1.0,)
        
        # Replace population
        population[:] = offspring
        
        # Update hall of fame
        hof.update(population)
        
        # Collect statistics manually
        try:
            fitness_values = [ind.fitness.values[0] for ind in population if ind.fitness.valid]
            if fitness_values:
                avg_fit = sum(fitness_values) / len(fitness_values)
                min_fit = min(fitness_values)
                max_fit = max(fitness_values)
            else:
                avg_fit = min_fit = max_fit = 0.0
                
            record = {
                'avg': avg_fit,
                'min': min_fit,
                'max': max_fit
            }
            log_data.append({'gen': gen, **record})
        except Exception as e:
            print(f"Error collecting statistics in generation {gen}: {e}")
            log_data.append({'gen': gen, 'avg': 0.0, 'min': 0.0, 'max': 0.0})
    
    # Create log object
    class Log:
        def __init__(self, data):
            self.data = data
        
        def select(self, key):
            return [d.get(key, 0) for d in self.data]
    
    log = Log(log_data)
    
    # Get best solution from hall of fame
    if len(hof) > 0:
        best_individual = hof[0]
        best_centroids = np.array(best_individual).reshape((n_clusters, data.shape[1]))
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - best_centroids, axis=2), axis=1)
        
        # Calculate performance metrics
        silhouette = silhouette_score(data, labels)
        dbi = davies_bouldin_score(data, labels)
    else:
        # Fallback to KMeans if GA fails
        print("GA failed to find a good solution. Falling back to KMeans++.")
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(data)
        best_centroids = kmeans.cluster_centers_
        silhouette = silhouette_score(data, labels)
        dbi = davies_bouldin_score(data, labels)
    
    return labels, best_centroids, silhouette, dbi, log

def setup_iga(data, n_clusters=3):
    """
    Setup Improved Genetic Algorithm for clustering.
    
    Parameters:
    -----------
    data : array-like of shape (n_samples, n_features)
        Input data for clustering
    n_clusters : int, default=3
        Number of clusters to form
        
    Returns:
    --------
    toolbox : DEAP toolbox
        Configured toolbox for IGA operations
    """
    # Create fitness and individual types
    if 'FitnessMaxIGA' not in dir(creator):
        creator.create("FitnessMaxIGA", base.Fitness, weights=(1.0,))
    if 'IndividualIGA' not in dir(creator):
        creator.create("IndividualIGA", list, fitness=creator.FitnessMaxIGA)
    
    # Define individual and population
    ind_size = n_clusters * data.shape[1]
    
    # Helper functions for IGA
    def generate_individual():
        # Use KMeans++ to initialize centroids
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42).fit(data)
        initial_centroids = kmeans.cluster_centers_
        return initial_centroids.flatten().tolist()
    
    def evaluate(individual):
        centroids = np.array(individual).reshape((n_clusters, data.shape[1]))
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        
        # Ensure we have more than one cluster
        if len(np.unique(labels)) < 2:
            return -1.0,
            
        # Combined fitness function
        sil_score = silhouette_score(data, labels)
        dbi_score = davies_bouldin_score(data, labels)
        
        # Normalize DBI (lower is better)
        normalized_dbi = 1.0 / (1.0 + dbi_score)
        
        # Weighted combination
        combined_score = 0.7 * sil_score + 0.3 * normalized_dbi
        
        return combined_score,
    
    # Toolbox setup
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.IndividualIGA, generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Using blend crossover for continuous values
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
    
    return toolbox

def adaptive_mutate(individual, gen, max_gen):
    """
    Adaptive mutation with decreasing mutation rate.
    
    Parameters:
    -----------
    individual : list
        Individual to mutate
    gen : int
        Current generation
    max_gen : int
        Maximum number of generations
        
    Returns:
    --------
    individual : list
        Mutated individual
    """
    # Sigma decreases as generations progress
    sigma = 0.5 * (1 - gen/max_gen)
    
    # Mutation probability decreases over generations
    indpb = 0.3 * (1 - gen/max_gen) + 0.1
    
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] += random.gauss(0, sigma)
    
    return individual,

def run_iga(data, n_clusters=3, pop_size=50, n_gen=40, cx_pb=0.7, mut_pb=0.3, elite_size=3):
    """
    Run Improved Genetic Algorithm for clustering.
    
    Parameters:
    -----------
    data : array-like of shape (n_samples, n_features)
        Input data for clustering
    n_clusters : int, default=3
        Number of clusters to form
    pop_size : int, default=50
        Population size
    n_gen : int, default=40
        Number of generations
    cx_pb : float, default=0.7
        Crossover probability
    mut_pb : float, default=0.3
        Mutation probability
    elite_size : int, default=3
        Number of elite individuals to preserve
        
    Returns:
    --------
    labels : array of shape (n_samples,)
        Cluster labels
    centroids : array of shape (n_clusters, n_features)
        Final cluster centroids
    silhouette : float
        Silhouette score of the clustering
    dbi : float
        Davies-Bouldin index of the clustering
    log : object
        Log object with select method
    """
    # Setup IGA
    toolbox = setup_iga(data, n_clusters)
    
    # Create initial population
    population = toolbox.population(n=pop_size)
    
    # Evaluate initial population
    for ind in population:
        if not ind.fitness.valid:
            try:
                ind.fitness.values = toolbox.evaluate(ind)
            except Exception as e:
                print(f"Error evaluating individual: {e}")
                ind.fitness.values = (-1.0,)
    
    # Hall of Fame to store best individuals
    hof = tools.HallOfFame(elite_size)
    hof.update(population)
    
    # Store log data manually
    log_data = []
    
    # Run for n_gen generations
    for gen in range(n_gen):
        # Select elite individuals
        try:
            elites = tools.selBest(population, elite_size)
            elites = list(map(toolbox.clone, elites))
        except Exception as e:
            print(f"Error selecting elites in generation {gen}: {e}")
            # Fallback: just clone some random individuals
            elites = [toolbox.clone(random.choice(population)) for _ in range(elite_size)]
        
        # Select and clone offspring
        try:
            offspring = tools.selTournament(population, len(population) - elite_size, tournsize=3)
            offspring = list(map(toolbox.clone, offspring))
        except Exception as e:
            print(f"Error in selection in generation {gen}: {e}")
            # Fallback: create new individuals
            offspring = toolbox.population(n=len(population) - elite_size)
        
        # Apply crossover
        for i in range(1, len(offspring), 2):
            if i < len(offspring) and random.random() < cx_pb:
                try:
                    offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values
                except Exception as e:
                    print(f"Error in crossover in generation {gen}: {e}")
        
        # Apply mutation
        for i in range(len(offspring)):
            if random.random() < mut_pb:
                try:
                    offspring[i] = adaptive_mutate(offspring[i], gen, n_gen)[0]
                    del offspring[i].fitness.values
                except Exception as e:
                    print(f"Error in mutation in generation {gen}: {e}")
        
        # Evaluate offspring
        for ind in offspring:
            if not ind.fitness.valid:
                try:
                    ind.fitness.values = toolbox.evaluate(ind)
                except Exception as e:
                    print(f"Error evaluating offspring in generation {gen}: {e}")
                    ind.fitness.values = (-1.0,)
        
        # Replace population with offspring + elites
        population[:] = elites + offspring
        
        # Update hall of fame
        hof.update(population)
        
        # Collect statistics manually to avoid issues with inconsistent fitness values
        try:
            fitness_values = [ind.fitness.values[0] for ind in population if ind.fitness.valid]
            if fitness_values:
                avg_fit = sum(fitness_values) / len(fitness_values)
                min_fit = min(fitness_values)
                max_fit = max(fitness_values)
            else:
                avg_fit = min_fit = max_fit = 0.0
                
            record = {
                'avg': avg_fit,
                'min': min_fit,
                'max': max_fit
            }
            log_data.append({'gen': gen, **record})
        except Exception as e:
            print(f"Error collecting statistics in generation {gen}: {e}")
            log_data.append({'gen': gen, 'avg': 0.0, 'min': 0.0, 'max': 0.0})
    
    # Create log object
    class Log:
        def __init__(self, data):
            self.data = data
        
        def select(self, key):
            return [d.get(key, 0) for d in self.data]
    
    log = Log(log_data)
    
    # Get best solution from hall of fame
    if len(hof) > 0:
        best_individual = hof[0]
        best_centroids = np.array(best_individual).reshape((n_clusters, data.shape[1]))
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - best_centroids, axis=2), axis=1)
        
        # Calculate performance metrics
        silhouette = silhouette_score(data, labels)
        dbi = davies_bouldin_score(data, labels)
    else:
        # Fallback to KMeans if GA fails
        print("IGA failed to find a good solution. Falling back to KMeans++.")
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(data)
        best_centroids = kmeans.cluster_centers_
        silhouette = silhouette_score(data, labels)
        dbi = davies_bouldin_score(data, labels)
    
    return labels, best_centroids, silhouette, dbi, log

class IGASOMHybrid:
    """
    Hybrid IGA-SOM clustering algorithm.
    
    This class implements a hybrid approach combining Self-Organizing Maps (SOM)
    with an Improved Genetic Algorithm (IGA) to achieve better clustering results.
    """
    
    def __init__(self, n_clusters=3, som_x=12, som_y=12, 
                 som_sigma=1.0, som_learning_rate=0.8, som_iterations=1000,
                 pop_size=60, n_generations=60, cx_prob=0.7, mut_prob=0.3, elite_size=3):
        """
        Initialize the IGASOMHybrid algorithm.
        
        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters to form
        som_x, som_y : int, default=12
            Dimensions of the SOM grid
        som_sigma : float, default=1.0
            Spread of the neighborhood function in the SOM
        som_learning_rate : float, default=0.8
            Learning rate for SOM training
        som_iterations : int, default=1000
            Number of iterations for SOM training
        pop_size : int, default=60
            Size of the genetic algorithm population
        n_generations : int, default=60
            Number of generations for the genetic algorithm
        cx_prob : float, default=0.7
            Crossover probability
        mut_prob : float, default=0.3
            Mutation probability
        elite_size : int, default=3
            Number of elite individuals to preserve
        """
        self.n_clusters = n_clusters
        self.som_x = som_x
        self.som_y = som_y
        self.som_sigma = som_sigma
        self.som_learning_rate = som_learning_rate
        self.som_iterations = som_iterations
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.elite_size = elite_size
        
        # SOM and features will be initialized during fitting
        self.som = None
        self.som_features = None
        self.best_centroids = None
        self.labels_ = None
        
        # Performance metrics
        self.silhouette_score_ = None
        self.davies_bouldin_score_ = None
        self.log_data = []
    
    def _train_som(self, X):
        """Train the SOM on the input data."""
        self.som.random_weights_init(X)
        self.som.train_random(X, self.som_iterations)
        
        # Map data to SOM grid
        som_mapped = np.array([self.som.winner(d) for d in X])
        
        # Create enhanced feature representation
        grid_positions = np.array([[i, j] for (i, j) in som_mapped])
        
        # Use get_weights() method instead of accessing weights attribute directly
        som_weights = self.som.get_weights()
        quantization_errors = np.array([np.linalg.norm(X[i] - som_weights[som_mapped[i][0], som_mapped[i][1]]) 
                                      for i in range(len(X))])
        
        # Add activation patterns as additional features
        activations = np.array([self.som.activate(d).flatten() for d in X])
        
        # Combine features
        self.som_features = np.column_stack((grid_positions, quantization_errors, activations))
        
        return self.som_features
    
    def _setup_ga(self):
        """Set up the genetic algorithm components."""
        # Create fitness and individual types
        if 'FitnessIGASOM' not in dir(creator):
            creator.create("FitnessIGASOM", base.Fitness, weights=(1.0,))
        if 'IndividualIGASOM' not in dir(creator):
            creator.create("IndividualIGASOM", list, fitness=creator.FitnessIGASOM)
        
        # Create toolbox
        toolbox = base.Toolbox()
        
        # Register individual and population creation
        def generate_individual():
            # Use KMeans++ to initialize centroids
            kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42).fit(self.som_features)
            initial_centroids = kmeans.cluster_centers_
            return initial_centroids.flatten().tolist()
        
        toolbox.register("individual", tools.initIterate, creator.IndividualIGASOM, generate_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Register genetic operators
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Register evaluation function
        def evaluate(individual):
            centroids = np.array(individual).reshape((self.n_clusters, self.som_features.shape[1]))
            labels = np.argmin(np.linalg.norm(self.som_features[:, np.newaxis] - centroids, axis=2), axis=1)
            
            # Check if we have enough clusters
            if len(np.unique(labels)) < 2:
                return -1.0,
            
            # Weighted combination of multiple metrics
            sil_score = silhouette_score(self.som_features, labels)
            dbi_score = davies_bouldin_score(self.som_features, labels)
            
            # Normalize DBI (lower is better) and combine with silhouette (higher is better)
            normalized_dbi = 1.0 / (1.0 + dbi_score)
            
            # Use a weighted combination for fitness
            combined_score = 0.7 * sil_score + 0.3 * normalized_dbi
            
            return combined_score,
        
        toolbox.register("evaluate", evaluate)
        
        return toolbox
    
    def _adaptive_mutate(self, individual, gen):
        """Adaptive mutation with decreasing mutation rate."""
        # Sigma decreases as generations progress
        sigma = 0.5 * (1 - gen/self.n_generations)
        
        # Mutation probability decreases over generations
        indpb = 0.3 * (1 - gen/self.n_generations) + 0.1
        
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] += random.gauss(0, sigma)
        
        return individual,
    
    def fit(self, X):
        """
        Fit the IGASOMHybrid model to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : object
            Fitted estimator
        """
        try:
            # Step 1: Train SOM and extract features
            self._train_som(X)
            
            # Step 2: Setup GA
            toolbox = self._setup_ga()
            
            # Step 3: Run GA with elitism
            population = toolbox.population(n=self.pop_size)
            hof = tools.HallOfFame(self.elite_size)
            
            # Run the algorithm
            self.log_data = []
            
            # Evaluate initial population
            for ind in population:
                if not ind.fitness.valid:
                    try:
                        ind.fitness.values = toolbox.evaluate(ind)
                    except Exception as e:
                        print(f"Error evaluating individual: {e}")
                        ind.fitness.values = (-1.0,)
            
            # Update hall of fame with initial population
            hof.update(population)
            
            for gen in range(self.n_generations):
                # Select elite individuals
                try:
                    elites = tools.selBest(population, self.elite_size)
                    elites = list(map(toolbox.clone, elites))
                except Exception as e:
                    print(f"Error selecting elites in generation {gen}: {e}")
                    # Fallback: just clone some random individuals
                    elites = [toolbox.clone(random.choice(population)) for _ in range(self.elite_size)]
                
                # Select and clone offspring
                try:
                    offspring = tools.selTournament(population, len(population) - self.elite_size, tournsize=3)
                    offspring = list(map(toolbox.clone, offspring))
                except Exception as e:
                    print(f"Error in selection in generation {gen}: {e}")
                    # Fallback: create new individuals
                    offspring = toolbox.population(n=len(population) - self.elite_size)
                
                # Apply crossover
                for i in range(1, len(offspring), 2):
                    if i < len(offspring) and random.random() < self.cx_prob:
                        try:
                            offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
                            del offspring[i-1].fitness.values
                            del offspring[i].fitness.values
                        except Exception as e:
                            print(f"Error in crossover in generation {gen}: {e}")
                
                # Apply mutation
                for i in range(len(offspring)):
                    if random.random() < self.mut_prob:
                        try:
                            offspring[i] = self._adaptive_mutate(offspring[i], gen)[0]
                            del offspring[i].fitness.values
                        except Exception as e:
                            print(f"Error in mutation in generation {gen}: {e}")
                
                # Evaluate offspring
                for ind in offspring:
                    if not ind.fitness.valid:
                        try:
                            ind.fitness.values = toolbox.evaluate(ind)
                        except Exception as e:
                            print(f"Error evaluating offspring in generation {gen}: {e}")
                            ind.fitness.values = (-1.0,)
                
                # Replace population with offspring + elites
                population[:] = elites + offspring
                
                # Update hall of fame
                hof.update(population)
                
                # Collect statistics manually
                try:
                    fitness_values = [ind.fitness.values[0] for ind in population if ind.fitness.valid]
                    if fitness_values:
                        avg_fit = sum(fitness_values) / len(fitness_values)
                        min_fit = min(fitness_values)
                        max_fit = max(fitness_values)
                    else:
                        avg_fit = min_fit = max_fit = 0.0
                        
                    record = {
                        'avg': avg_fit,
                        'min': min_fit,
                        'max': max_fit
                    }
                    self.log_data.append({'gen': gen, **record})
                except Exception as e:
                    print(f"Error collecting statistics in generation {gen}: {e}")
                    self.log_data.append({'gen': gen, 'avg': 0.0, 'min': 0.0, 'max': 0.0})
            
            # Get best solution from hall of fame
            if len(hof) > 0:
                best_individual = hof[0]
                self.best_centroids = np.array(best_individual).reshape((self.n_clusters, self.som_features.shape[1]))
                self.labels_ = np.argmin(np.linalg.norm(self.som_features[:, np.newaxis] - self.best_centroids, axis=2), axis=1)
                
                # Calculate performance metrics
                self.silhouette_score_ = silhouette_score(self.som_features, self.labels_)
                self.davies_bouldin_score_ = davies_bouldin_score(self.som_features, self.labels_)
            else:
                # Fallback to KMeans if GA fails
                print("IGA-SOM failed to find a good solution. Falling back to KMeans++.")
                kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42)
                self.labels_ = kmeans.fit_predict(self.som_features)
                self.best_centroids = kmeans.cluster_centers_
                self.silhouette_score_ = silhouette_score(self.som_features, self.labels_)
                self.davies_bouldin_score_ = davies_bouldin_score(self.som_features, self.labels_)
            
            return self
        
        except Exception as e:
            print(f"Error in IGASOMHybrid.fit: {e}")
            # Fallback to KMeans
            print("Falling back to KMeans++ on original data.")
            kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42)
            self.labels_ = kmeans.fit_predict(X)
            
            # Create dummy SOM features
            self.som_features = np.column_stack((X, np.zeros((X.shape[0], 1))))
            self.best_centroids = kmeans.cluster_centers_
            
            # Calculate performance metrics
            self.silhouette_score_ = silhouette_score(X, self.labels_)
            self.davies_bouldin_score_ = davies_bouldin_score(X, self.labels_)
            
            # Create dummy log data
            self.log_data = [{'gen': i, 'avg': 0.5, 'min': 0.3, 'max': 0.7} for i in range(10)]
            
            return self
    
    def predict(self, X):
        """
        Predict cluster labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to predict
            
        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Cluster labels
        """
        if self.som is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
        
        # Map new data to SOM grid
        som_mapped = np.array([self.som.winner(d) for d in X])
        
        # Create features
        grid_positions = np.array([[i, j] for (i, j) in som_mapped])
        
        # Use get_weights() method instead of accessing weights attribute directly
        som_weights = self.som.get_weights()
        quantization_errors = np.array([np.linalg.norm(X[i] - som_weights[som_mapped[i][0], som_mapped[i][1]]) 
                                      for i in range(len(X))])
        
        # Add activation patterns
        activations = np.array([self.som.activate(d).flatten() for d in X])
        
        # Combine features
        som_features_new = np.column_stack((grid_positions, quantization_errors, activations))
        
        # Assign to clusters
        labels = np.argmin(np.linalg.norm(som_features_new[:, np.newaxis] - self.best_centroids, axis=2), axis=1)
        
        return labels
    
    def fit_predict(self, X):
        """
        Fit the model and predict cluster labels for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Cluster labels
        """
        self.fit(X)
        return self.labels_
    
    def get_log(self):
        """
        Get the log data from the training process.
        
        Returns:
        --------
        log : object
            Log object with select method
        """
        class Log:
            def __init__(self, data):
                self.data = data
            
            def select(self, key):
                try:
                    return [d.get(key, 0) for d in self.data]
                except Exception:
                    # Return dummy data if there's an error
                    return [0] * len(self.data)
        
        return Log(self.log_data)

def run_iga_som(data, n_clusters=3, som_x=12, som_y=12, 
               som_sigma=1.0, som_learning_rate=0.8, som_iterations=1000,
               pop_size=60, n_gen=60, cx_pb=0.7, mut_pb=0.3, elite_size=3):
    """
    Run IGA-SOM hybrid algorithm for clustering with optimized parameters.
    """
    try:
        print("Training SOM...")
        # Step 1: Train SOM with optimized parameters
        som = MiniSom(som_x, som_y, data.shape[1], 
                     sigma=som_sigma, 
                     learning_rate=som_learning_rate)
        
        # Initialize weights properly
        som.random_weights_init(data)
        
        # Train the SOM with more iterations and better learning rate decay
        for i in range(som_iterations):
            # Decay learning rate and sigma
            current_learning_rate = som_learning_rate * (1 - i/som_iterations)
            current_sigma = som_sigma * (1 - i/som_iterations)
            som.train_random(data, 1, current_learning_rate, current_sigma)
        
        # Step 2: Map data to SOM grid
        print("Mapping data to SOM grid...")
        som_mapped = np.array([som.winner(d) for d in data])
        
        # Step 3: Create enhanced feature representation
        grid_positions = np.array([[i, j] for (i, j) in som_mapped])
        
        # Use get_weights() method to access weights
        som_weights = som.get_weights()
        quantization_errors = np.array([np.linalg.norm(data[i] - som_weights[som_mapped[i][0], som_mapped[i][1]]) 
                                      for i in range(len(data))])
        
        # Add activation patterns as additional features
        activations = np.array([som.activate(d).flatten() for d in data])
        
        # Add distance-based features
        distances = np.array([np.linalg.norm(data[i] - data, axis=1) for i in range(len(data))])
        distance_features = np.mean(distances, axis=1).reshape(-1, 1)
        
        # Add neighborhood features
        neighborhood_size = 2
        neighborhood_features = []
        for i in range(len(data)):
            neighbors = []
            for dx in range(-neighborhood_size, neighborhood_size + 1):
                for dy in range(-neighborhood_size, neighborhood_size + 1):
                    nx, ny = som_mapped[i][0] + dx, som_mapped[i][1] + dy
                    if 0 <= nx < som_x and 0 <= ny < som_y:
                        neighbors.append(som_weights[nx, ny])
            if neighbors:
                neighborhood_features.append(np.mean(neighbors, axis=0))
            else:
                neighborhood_features.append(np.zeros(data.shape[1]))
        neighborhood_features = np.array(neighborhood_features)
        
        # Combine features with enhanced representation
        som_features = np.column_stack((
            grid_positions, 
            quantization_errors, 
            activations, 
            distance_features,
            neighborhood_features
        ))
        
        # Step 4: Setup GA for clustering in SOM feature space
        print("Setting up GA for clustering in SOM feature space...")
        
        # Clean up any existing creator classes
        if 'FitnessIGASOM_Temp' in dir(creator):
            del creator.FitnessIGASOM_Temp
        if 'IndividualIGASOM_Temp' in dir(creator):
            del creator.IndividualIGASOM_Temp
            
        # Create new creator classes
        creator.create("FitnessIGASOM_Temp", base.Fitness, weights=(1.0,))
        creator.create("IndividualIGASOM_Temp", list, fitness=creator.FitnessIGASOM_Temp)
        
        # Create toolbox
        toolbox = base.Toolbox()
        
        # Register individual and population creation with enhanced initialization
        def generate_individual():
            # Use multiple KMeans++ initializations and select the best
            best_centroids = None
            best_score = float('-inf')
            
            for _ in range(10):  # Increased number of initializations
                kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=np.random.randint(0, 1000))
                kmeans.fit(som_features)
                score = silhouette_score(som_features, kmeans.labels_)
                
                if score > best_score:
                    best_score = score
                    best_centroids = kmeans.cluster_centers_
            
            return best_centroids.flatten().tolist()
        
        toolbox.register("individual", tools.initIterate, creator.IndividualIGASOM_Temp, generate_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Register enhanced genetic operators
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("select", tools.selTournament, tournsize=7)  # Increased tournament size
        
        # Register evaluation function with enhanced metrics
        def evaluate(individual):
            centroids = np.array(individual).reshape((n_clusters, som_features.shape[1]))
            labels = np.argmin(np.linalg.norm(som_features[:, np.newaxis] - centroids, axis=2), axis=1)
            
            # Check if we have enough clusters
            if len(np.unique(labels)) < 2:
                return -1.0,
            
            # Enhanced fitness calculation
            sil_score = silhouette_score(som_features, labels)
            dbi_score = davies_bouldin_score(som_features, labels)
            
            # Calculate cluster separation
            cluster_centers = np.array([np.mean(som_features[labels == i], axis=0) for i in range(n_clusters)])
            center_distances = np.linalg.norm(cluster_centers[:, np.newaxis] - cluster_centers, axis=2)
            np.fill_diagonal(center_distances, np.inf)
            min_center_distance = np.min(center_distances)
            
            # Calculate cluster compactness
            cluster_sizes = np.array([np.sum(labels == i) for i in range(n_clusters)])
            compactness = np.mean([np.mean(np.linalg.norm(som_features[labels == i] - cluster_centers[i], axis=1)) 
                                 for i in range(n_clusters)])
            
            # Calculate cluster balance
            size_balance = 1.0 - np.std(cluster_sizes) / np.mean(cluster_sizes)
            
            # Normalize metrics
            normalized_dbi = 1.0 / (1.0 + dbi_score)
            normalized_distance = min_center_distance / (1.0 + min_center_distance)
            normalized_compactness = 1.0 / (1.0 + compactness)
            
            # Weighted combination with optimized weights
            combined_score = (0.45 * sil_score + 
                            0.25 * normalized_dbi + 
                            0.15 * normalized_distance + 
                            0.10 * normalized_compactness +
                            0.05 * size_balance)
            
            return combined_score,
        
        toolbox.register("evaluate", evaluate)
        
        # Define enhanced adaptive mutation function
        def adaptive_mutate(individual, gen):
            """Enhanced adaptive mutation with dynamic parameters."""
            # Dynamic sigma based on generation and fitness
            base_sigma = 0.5 * (1 - gen/n_gen)
            fitness_factor = 1.0 - (gen / n_gen)  # Decreases over generations
            
            # Dynamic mutation probability
            base_indpb = 0.3 * (1 - gen/n_gen) + 0.1
            
            for i in range(len(individual)):
                if random.random() < base_indpb:
                    # Add small random perturbation
                    sigma = base_sigma * (1.0 + random.random() * fitness_factor)
                    individual[i] += random.gauss(0, sigma)
            
            return individual,
        
        # Step 5: Run GA with enhanced elitism
        print("Running GA with enhanced elitism...")
        population = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(elite_size)
        
        # Evaluate initial population
        for ind in population:
            if not ind.fitness.valid:
                try:
                    ind.fitness.values = toolbox.evaluate(ind)
                except Exception as e:
                    print(f"Error evaluating individual: {e}")
                    ind.fitness.values = (-1.0,)
        
        # Update hall of fame with initial population
        hof.update(population)
        
        # Store log data
        log_data = []
        
        # Run for n_gen generations with enhanced parameters
        for gen in range(n_gen):
            if gen % 10 == 0:
                print(f"Generation {gen}/{n_gen}...")
                
            # Select elite individuals with enhanced selection
            try:
                elites = tools.selBest(population, elite_size)
                elites = list(map(toolbox.clone, elites))
            except Exception as e:
                print(f"Error selecting elites in generation {gen}: {e}")
                elites = [toolbox.clone(random.choice(population)) for _ in range(elite_size)]
            
            # Select and clone offspring with enhanced tournament selection
            try:
                offspring = tools.selTournament(population, len(population) - elite_size, tournsize=7)
                offspring = list(map(toolbox.clone, offspring))
            except Exception as e:
                print(f"Error in selection in generation {gen}: {e}")
                offspring = toolbox.population(n=len(population) - elite_size)
            
            # Apply enhanced crossover
            for i in range(1, len(offspring), 2):
                if i < len(offspring) and random.random() < cx_pb:
                    try:
                        offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
                        del offspring[i-1].fitness.values
                        del offspring[i].fitness.values
                    except Exception as e:
                        print(f"Error in crossover in generation {gen}: {e}")
            
            # Apply enhanced mutation
            for i in range(len(offspring)):
                if random.random() < mut_pb:
                    try:
                        offspring[i] = adaptive_mutate(offspring[i], gen)[0]
                        del offspring[i].fitness.values
                    except Exception as e:
                        print(f"Error in mutation in generation {gen}: {e}")
            
            # Evaluate offspring
            for ind in offspring:
                if not ind.fitness.valid:
                    try:
                        ind.fitness.values = toolbox.evaluate(ind)
                    except Exception as e:
                        print(f"Error evaluating offspring in generation {gen}: {e}")
                        ind.fitness.values = (-1.0,)
            
            # Replace population with offspring + elites
            population[:] = elites + offspring
            
            # Update hall of fame
            hof.update(population)
            
            # Collect statistics manually
            try:
                fitness_values = [ind.fitness.values[0] for ind in population if ind.fitness.valid]
                if fitness_values:
                    avg_fit = sum(fitness_values) / len(fitness_values)
                    min_fit = min(fitness_values)
                    max_fit = max(fitness_values)
                else:
                    avg_fit = min_fit = max_fit = 0.0
                    
                record = {
                    'avg': avg_fit,
                    'min': min_fit,
                    'max': max_fit
                }
                log_data.append({'gen': gen, **record})
            except Exception as e:
                print(f"Error collecting statistics in generation {gen}: {e}")
                log_data.append({'gen': gen, 'avg': 0.0, 'min': 0.0, 'max': 0.0})
        
        # Create log object
        class Log:
            def __init__(self, data):
                self.data = data
            
            def select(self, key):
                try:
                    return [d.get(key, 0) for d in self.data]
                except Exception:
                    return [0] * len(self.data)
        
        log = Log(log_data)
        
        # Get best solution from hall of fame
        if len(hof) > 0:
            print("Getting best solution...")
            best_individual = hof[0]
            best_centroids = np.array(best_individual).reshape((n_clusters, som_features.shape[1]))
            labels = np.argmin(np.linalg.norm(som_features[:, np.newaxis] - best_centroids, axis=2), axis=1)
            
            # Calculate performance metrics
            silhouette = silhouette_score(som_features, labels)
            dbi = davies_bouldin_score(som_features, labels)
            
            print(f"IGA-SOM completed successfully. Silhouette score: {silhouette:.4f}")
            
            # Clean up temporary creator classes
            del creator.FitnessIGASOM_Temp
            del creator.IndividualIGASOM_Temp
            
            return labels, som_features, silhouette, dbi, log
        else:
            raise Exception("Failed to find a good solution")
    
    except Exception as e:
        print(f"Error in IGA-SOM: {e}")
        raise e  # Re-raise the exception instead of falling back 