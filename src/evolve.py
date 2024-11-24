"""
An implementation of the Microbial Genetic Algorithm for the DGCA SLP weights, 
including the use of separate "chromosomes" as described in the paper.
"""
from __future__ import annotations
from collections.abc import Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from graph_tool.all import Graph
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()
from .dgca import DGCA, Runner, GraphDef

def rmse(p: np.ndarray, q: np.ndarray) -> float:
    return np.sqrt(np.mean(np.power(p - q, 2)))

def mean_absolute_error(p: np.ndarray, q: np.ndarray) -> float:
    return np.mean(np.abs(p - q))

def canberra_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Distance between two vectors. 
    https://en.wikipedia.org/wiki/Canberra_distance
    Used by @shellmanMetabolicNetworkMotifs2014 to compare motif
    significance profile similarity.
    Maybe I should use this rather than MAE.
    Also available as
    >>> from scipy.spatial.distance import canberra
    """
    return np.sum(np.abs(p-q) / (np.abs(p) + np.abs(q)))

@dataclass
class Chromosome:
    """
    Holds the data for a "chromosome". Multiple chromosomes make up a genome.
    In this case the Chromosome just holds SLP weights. It also implements
    mutation and crosssover methods.
    """

    data: np.ndaray
    mutate_rate: float
    crossover_rate: float
    crossover_style: str
    best_fitness: np.float32 = np.nan

    def mutate(self) -> Chromosome:
        mask = np.random.choice([True, False], p=[self.mutate_rate, 1-self.mutate_rate], size=self.data.shape)
        random = np.random.uniform(size=self.data.shape, low=-1, high=1).astype(np.float32)
        self.data[mask] = random[mask]
        self.best_fitness = np.nan
        return self # to enable chained calls
        #return Chromosome(newdata, self.mutate_rate, self.crossover_rate, self.crossover_style)
    
    def crossover(self, other: Chromosome) -> Chromosome:
        assert self.data.shape == other.data.shape, "Mismatched shapes for Chromosome.data"
        n_rows, n_cols = self.data.shape
        if self.crossover_style =='rows':
            row_idx = np.random.choice(range(n_rows), 
                                       size=(round(n_rows * self.crossover_rate),), 
                                       replace=False)
            self.data[row_idx,:] = other.data[row_idx,:]
        elif self.crossover_style =='cols':
            col_idx = np.random.choice(range(n_cols), 
                                       size=(round(n_cols * self.crossover_rate),), 
                                       replace=False)
            self.data[:,col_idx] = other.data[:,col_idx]
        else:
            raise ValueError("crossover_style should be 'rows' or 'cols'")
        self.best_fitness = np.nan
        return self # to enable chained calls
        #return Chromosome(newdata, self.mutate_rate, self.crossover_rate, self.crossover_style)
    
    def get_new(self) -> Chromosome:
        """
        Factory method. Creates a copy of this object with same attribs except data.
        Equivalent to a 100% mutation.
        """
        newdata = np.random.uniform(size=self.data.shape, low=-1, high=1).astype(np.float32)
        return Chromosome(newdata, self.mutate_rate, self.crossover_rate, self.crossover_style, best_fitness=np.nan)
    

class EvolvableDGCA(DGCA):

    def set_chromosomes(self, chr1: Chromosome, chr2: Chromosome) -> None:
        assert self.action_slp_wgts.shape == chr1.data.shape, "Shape mismatch for chromosome 1"
        assert self.state_slp_wgts.shape == chr2.data.shape, "Shape mismatch for chromosome 2"
        self.action_slp_wgts = chr1.data
        self.state_slp_wgts = chr2.data

    def get_chromosomes(self, mutate_rate: float, cross_rate: float, cross_style: str) -> list[Chromosome]:
        chr1 = Chromosome(self.action_slp_wgts.copy(), mutate_rate, cross_rate, cross_style)
        chr2 = Chromosome(self.state_slp_wgts.copy(), mutate_rate, cross_rate, cross_style)
        return chr1, chr2


class GraphFitness(ABC):
    """
    Abstract Base Class / Interface for a Callable which takes a GraphDef and returns
    its fitnesss
    """
    def __init__(self, high_good: bool):
        self.high_good = high_good
        self.skip_count = 0

    @abstractmethod
    def __call__(self, graph: GraphDef) -> tuple[float,...]:
        raise NotImplementedError


def check_conditions(gd: GraphDef, conditions: dict[str, float], 
                     verbose: bool = False) -> bool:
    size = gd.size()
    conn = gd.connectivity()
    frag = gd.get_largest_component_frac()
    if 'max_size' in conditions:
        # should already be fine but double check
        if gd.size() > conditions['max_size']:
            if verbose:
                print('Graph too big (should not happen!)')
            return False
    if 'min_size' in conditions:
        if gd.size() < conditions['min_size']:
            if verbose:
                print('Graph too small.')
            return False
    if 'min_connectivity' in conditions:
        if gd.connectivity() < conditions['min_connectivity']:
            if verbose:
                print('Graph too sparse')
            return False
    if 'max_connectivity' in conditions:
        if gd.connectivity() > conditions['max_connectivity']:
            if verbose:
                print('Graph too dense')
            return False
    if 'min_component_frac' in conditions:
        if gd.get_largest_component_frac() < conditions['min_component_frac']:
            if verbose:
                print('Graph too fragmented')
            return False
    if verbose:
        print(f'Graph OK: size={size}, conn={conn*100:.2f}%, frag={frag:.2f}')
    return True


class SignificanceProfileFitness(GraphFitness):
    """

    """
    def __init__(self,
                 target_sig_prof: np.ndarray,
                 zscore_func: Callable[[Graph], tuple],
                 err_func: Callable[[np.ndarray, np.ndarray], float] = mean_absolute_error,
                 self_loops: bool = False,
                 conditions: dict = dict(),
                 verbose: bool = False
                 ) -> None:
        """
        zscore func should be a partial() of gt.motif_significance with params set
        """
        super().__init__(high_good = False) # because we will be returning an error metric, so low is good.
        self.target_sig_prof = target_sig_prof
        self.zscore_func = zscore_func
        self.err_func = err_func
        self.self_loops = self_loops
        self.conditions = conditions
        self.verbose = verbose
        self.high_good = False # because we will return mean abs error
        self.memo = {'fitness':[], 'sig_prof':[], 'graph':[], 'model':[]}


    def __call__(self, graph: GraphDef) -> float:
        if not self.self_loops:
            graph = graph.no_selfloops()
        checks_ok = check_conditions(graph,self.conditions)
        if checks_ok:
            if self.verbose:
                print('Calculating motif significance')
            res = self.zscore_func(graph.to_gt(basic=True))
            zscores = np.array(res[1])
            sig_prof = zscores / np.linalg.norm(zscores, keepdims=True)
            err = self.err_func(sig_prof, self.target_sig_prof)
            if self.verbose:
                print(f'Skipped {self.skip_count}')
            self.skip_count = 0
        else:
            self.skip_count += 1
            err = np.nan
        return err, zscores


class ChromosomalMGA:

    def __init__(self, popsize: int,
                  model: EvolvableDGCA,
                  seed_graph: GraphDef,
                  runner: Runner,
                  fitness_fn: GraphFitness,
                  mutate_rate: float, cross_rate: float, cross_style: str,
                  csv_filename: str | None = None):
        self.popsize = popsize
        self.model = model
        self.seed_graph = seed_graph
        self.runner = runner
        self.fitness_fn = fitness_fn
        self.csv_filename = csv_filename
        # Comparison funcs are nan tolerant
        if self.fitness_fn.high_good:
            self.better = lambda a, b: np.isnan(b) or a>=b
        else:
            self.better = lambda a, b: np.isnan(b) or a<=b
        self.base_chromosomes = self.model.get_chromosomes(mutate_rate, cross_rate, cross_style)
        self.pop_chromosomes = np.array([[bc.get_new() for _ in range(self.popsize)] for bc in self.base_chromosomes]).T
        self.num_chromosomes = len(self.base_chromosomes)
        self.fitness_record = [] # 
        if self.csv_filename is not None:
            print(f'CSV results will be written to: {self.csv_filename}')

    def run(self, steps: int) -> list[float]:
        pbar = tqdm(range(steps),postfix={'fit':0,'best':0})
        for s in pbar:
            f = self.contest()
            best_fitness = np.max(self.records['fitness']) if self.fit_fn.high_good else np.min(self.records['fitness'])
            pbar.set_postfix({'fit':f,'best':best_fitness})
        return self.records['fitness']

    def contest(self) -> float:
        """
        Runs a single contest between two individuals created out of randomly selected chromosomes
        Returns the fitness of the fitter one.
        """
        # Select two sets of chromosomes at random
        idx = np.random.randint(low=0,high=self.popsize,size=(2,self.num_chromosomes))
        contestant_chromosomes : list[list[Chromosome]] = np.take_along_axis(self.pop_chromosomes, idx, axis=0).tolist()
        fitness = (np.nan, np.nan)
        while np.all(np.isnan(fitness)):
            fitness = self.run_individual(contestant_chromosomes[0]), self.run_individual(contestant_chromosomes[1])
            # If both contestants' fitness is NaN then mutate the chromosomes
            # and try again (in this while loop so that it deosn't count as an iteration in run())
            if np.all(np.isnan(fitness)):
                for chr in contestant_chromosomes.flat:
                    if not np.isnan(chr.best_fitness):
                        chr.mutate() 
        win, lose = 0,1 if self.better(*fitness) else 1,0
        for c in range(self.num_chromosomes):
            if idx[win,c]==idx[lose,c]:
                # These two chromosomes were the same, so don't change anything
                continue
            else:
                chr_win, chr_lose = contestant_chromosomes[win,c], contestant_chromosomes[lose,c]
                # Don't change the losing individual if it has previously been part of an 
                # individual with higher fitness than the winner
                if self.better(fitness[win], chr_lose.best_fitness):
                    #call crossover & mutate on the loser (this changes it in place)
                    chr_lose.crossover(chr_win).mutate()
        return fitness[win]
    
    def run_individual(self, chromosomes: list[Chromosome]) -> float:
        """
        Returns the fitness of one set of chromosomes.
        """
        self.model.set_chromosomes(*chromosomes)
        self.runner.reset()
        final_graph = self.runner.run(self.model, self.seed_graph)
        fitness, info = self.fitness_fn(final_graph)
        # Update each chromosome's best_fitness score
        for chr in chromosomes:
            if self.better(fitness, chr.best_fitness):
                chr.best_fitness = fitness
        if np.isnan(fitness) and (self.csv_filename is not None):
            # Save some stuff.
            # self.memo['fitness'].append(fitness)
            # self.memo['sig_prof'].append(info)
            # self.memo['graph'].append(final_graph)
            # self.memo['model'].append(jsonpickle.encode(self.model))
            # New: save csv row of MAE,zscore1,zscroe2,... (zscores not sig prof so that we can calc |z|)
            csv_row = (f'{fitness:.5f}\t'
                    +'\t'.join(['%.5f' % num for num in info])
                    +'\t'+jsonpickle.encode(self.model)
                    +'\t'+jsonpickle.encode(final_graph)
                    +'\t'+f'{self.fitness_fn.skip_count}' # so that we can keep track of how many we are skipping.
                    +'\n')
            with open(self.csv_filename,'a') as fh:
                fh.write(csv_row)
                fh.flush() # I think this updates it immediately (good if we want to read as we go along)
        return fitness
