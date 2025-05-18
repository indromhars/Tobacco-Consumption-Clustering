"""
Improved IGA-SOM Hybrid Algorithm for Tobacco Consumption Clustering

This module implements an enhanced hybrid algorithm combining Improved Genetic Algorithm (IGA)
with Self-Organizing Maps (SOM) for better clustering performance.
"""

import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from minisom import MiniSom
from deap import base, creator, tools, algorithms
import pickle

class ImprovedIGASOM:
    """
    Improved IGA-SOM hybrid clustering algorithm.
    
    This class implements a hybrid approach combining Self-Organizing Maps (SOM)
    with an Improved Genetic Algorithm (IGA) to achieve better clustering results.
    """
    
    def __init__(self, n_clusters=3, som_x=12, som_y=12, 
                 som_sigma=1.0, som_learning_rate=0.8, som_iterations=1000,
                 pop_size=60, n_generations=60, cx_prob=0.7, mut_prob=0.3, elite_size=3):
        """
        Initialize the ImprovedIGASOM algorithm.
        
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
        
    def _train_som(self, X):
        """Train the SOM and extract features."""
        # Initialize and train SOM
        self.som = MiniSom(self.som_x, self.som_y, X.shape[1], 
                          sigma=self.som_sigma, 
                          learning_rate=self.som_learning_rate)
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
        # Use PCA to reduce activation dimensions
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        activation_features = pca.fit_transform(activations)
        
        # Combine features
        self.som_features = np.column_stack((grid_positions, quantization_errors, activation_features))
        
        return self.som_features
    
    def _setup_ga(self):
        """Set up the genetic algorithm components."""
        # Create fitness and individual types
        if 'FitnessEnhanced' not in dir(creator):
            creator.create("FitnessEnhanced", base.Fitness, weights=(1.0,))
        if 'IndividualEnhanced' not in dir(creator):
            creator.create("IndividualEnhanced", list, fitness=creator.FitnessEnhanced)
        
        # Create toolbox
        toolbox = base.Toolbox()
        
        # Register individual and population creation
        def generate_individual():
            # Use KMeans++ to initialize centroids
            kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42).fit(self.som_features)
            initial_centroids = kmeans.cluster_centers_
            return initial_centroids.flatten().tolist()
        
        toolbox.register("individual", tools.initIterate, creator.IndividualEnhanced, generate_individual)
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
        Fit the ImprovedIGASOM model to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : object
            Fitted estimator
        """
        # Step 1: Train SOM and extract features
        self._train_som(X)
        
        # Step 2: Setup GA
        toolbox = self._setup_ga()
        
        # Step 3: Run GA with elitism
        population = toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(self.elite_size)
        
        # Statistics tracking
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run the algorithm
        for gen in range(self.n_generations):
            # Select elite individuals
            elites = tools.selBest(population, self.elite_size)
            elites = list(map(toolbox.clone, elites))
            
            # Select and clone offspring
            offspring = tools.selTournament(population, len(population) - self.elite_size, tournsize=3)
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover
            for i in range(1, len(offspring), 2):
                if random.random() < self.cx_prob:
                    offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values
            
            # Apply mutation
            for i in range(len(offspring)):
                if random.random() < self.mut_prob:
                    offspring[i] = self._adaptive_mutate(offspring[i], gen)[0]
                    del offspring[i].fitness.values
            
            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population with offspring + elites
            population[:] = elites + offspring
            
            # Update hall of fame
            hof.update(population)
        
        # Get best solution
        best_individual = hof[0]
        self.best_centroids = np.array(best_individual).reshape((self.n_clusters, self.som_features.shape[1]))
        self.labels_ = np.argmin(np.linalg.norm(self.som_features[:, np.newaxis] - self.best_centroids, axis=2), axis=1)
        
        # Calculate performance metrics
        self.silhouette_score_ = silhouette_score(self.som_features, self.labels_)
        self.davies_bouldin_score_ = davies_bouldin_score(self.som_features, self.labels_)
        
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
        # Use PCA to reduce activation dimensions
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        activation_features = pca.fit_transform(activations)
        
        # Combine features
        som_features_new = np.column_stack((grid_positions, quantization_errors, activation_features))
        
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

# Save the best model and results
import pickle

# Create a dictionary with all models and results
models = {
    'kmeans': {
        'labels': kmeans_labels,
        'centroids': kmeans.cluster_centers_,
        'silhouette': kmeans_silhouette,
        'dbi': kmeans_dbi
    },
    'ga': {
        'labels': ga_labels,
        'centroids': ga_centroids,
        'silhouette': ga_silhouette,
        'dbi': ga_dbi
    },
    'iga': {
        'labels': iga_labels,
        'centroids': iga_centroids,
        'silhouette': iga_silhouette,
        'dbi': iga_dbi
    },
    'iga_som': {
        'labels': iga_som_labels,
        'silhouette': iga_som_silhouette,
        'dbi': iga_som_dbi
    },
    'results_df': results
}

# Save models and results
with open('clustering_results.pkl', 'wb') as f:
    pickle.dump(models, f)

print("Models and results saved to 'clustering_results.pkl'") 