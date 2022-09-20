from generate_poised_sets import SampleSets
from lagrange_polynomial import LagrangePolynomials
import numpy as np

if __name__ == '__main__':
    
    sets = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]).T
    
    SampleSets(sets)