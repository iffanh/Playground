from generate_poised_sets import SampleSets
from lagrange_polynomial import LagrangePolynomials
import numpy as np

if __name__ == '__main__':
    
    sets = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]).T
    
    ball = SampleSets(sets).ball
    print(f"Center : {ball.center}")
    print(f"Radius : {ball.rad}")
    
    N_points = 5
    vectors = ball.generate_vectors_with_uniform_angles(N_points)
    print(f"{N_points} vectors forming uniform angles = {vectors}")