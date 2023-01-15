from .lagrange_polynomial import LagrangePolynomials, LagrangePolynomial
import numpy as np
from typing import Any, Tuple, List

class ModelImprovement:
    """ Class that responsible for improving the lagrange polynomial models based on the poisedness of set Y. 
    
    """
    def __init__(self, input_symbols) -> None:
        self.input_symbols = input_symbols
        pass
    
    def improve_model(self, lpolynomials:LagrangePolynomials, rad:float, center:np.ndarray, L:float=1.0, max_iter:int=5, sort_type='function') -> Tuple[LagrangePolynomials, dict]:
        """ The function responsible for improving the poisedness of set Y in lagrange polynomial. 
        It follows from Algorithm 6.3 in Conn's book.

        Args:
            lpolynomials (LagrangePolynomials): LagrangePolynomials object to be improved
            func (callable): function call to evaluate the new points
            L (float, optional): Maximum poisedness in the new LagrangePolynomial object. Defaults to 100.0.
            max_iter (int, optional): Number of loops to get to the improved poisedness. Defaults to 5.

        Returns:
            LagrangePolynomials: New LagrangePolynomial object with improved poisedness
        """
        
        for k in range(max_iter):
            # Algorithm 6.3
            poisedness = lpolynomials.poisedness(rad=rad, center=center)
            Lambda = poisedness.max_poisedness()


            ## TODO: Any ideas on how to circumvent the replacement of the best point?
            pindex = poisedness.index
            if pindex == 0:
                
                print(f"Rebuild sample points")
                
                tr_radius = lpolynomials.tr_radius*1
                new_y = lpolynomials.y*1
                Y_remaining = lpolynomials.y[:, 1:]
                shifts = lpolynomials.y[:, [0]] - np.mean(Y_remaining, axis=1)[:, np.newaxis]
                new_y[:, 1:] = lpolynomials.y[:, 1:] + shifts

                for i in range(1, new_y.shape[1]):
                    new_y[:,i] = new_y[:,0] + (new_y[:,i] - new_y[:,0])*tr_radius/np.linalg.norm(new_y[:,i] - new_y[:,0])

                lpolynomials = LagrangePolynomials(input_symbols=self.input_symbols, pdegree=2)
                lpolynomials.initialize(v=new_y, f=None, sort_type=sort_type, tr_radius=tr_radius)   

                best_polynomial = lpolynomials
                curr_Lambda = Lambda*1
                break


            else:
                
                if k == 0:
                    best_polynomial = lpolynomials
                    curr_Lambda = Lambda*1

                # main loop
                if Lambda > L:
                    # find new point and its OF
                    new_point = poisedness.point_to_max_poisedness()
                    
                    is_redundant = False
                    for ii in range(lpolynomials.y.shape[1]):
                        if (new_point == lpolynomials.y[:,ii]).all():
                            is_redundant = True
                            break
                    
                    if is_redundant:
                        break
                    
                    is_new_point_a_duplicate = False
                    for i in range(lpolynomials.y.shape[1]):
                        if (new_point == lpolynomials.y[:,i]).all():
                            print("Point already exist")
                            is_new_point_a_duplicate = True
                            break
                    if is_new_point_a_duplicate:
                        break

                    # copy values
                    new_y = lpolynomials.y*1
                    tr_radius = lpolynomials.tr_radius*1
                    
                    # replace value
                    new_y[:, pindex] = new_point
                    
                    # create polynomials
                    lpolynomials = LagrangePolynomials(input_symbols=self.input_symbols, pdegree=2)
                    lpolynomials.initialize(v=new_y, f=None, sort_type=sort_type, tr_radius=tr_radius)       
                    
                    # save polynomial with the smallest poisedness
                    if Lambda < curr_Lambda:

                        curr_Lambda = Lambda*1
                        best_polynomial = LagrangePolynomials(input_symbols=self.input_symbols, pdegree=2)
                        
                        best_polynomial.initialize(v=new_y, f=None, sort_type=sort_type, tr_radius=tr_radius)
                        
                        if curr_Lambda < L:
                            return best_polynomial
                else:
                    
                    # Ad-hoc:
                    # - check if the number of points given the radius is enough for interpolation.
                    # - if not, scale the outside points with the said radius
                    # - should have some look up table to see whether points inside are available
                    
                    ## TODO: Maybe algorithm 6.2
                        
                    # break
                    rad_ratio = rad/lpolynomials.sample_set.ball.rad
                    if rad_ratio < 1.0:
                        new_y = (lpolynomials.y - lpolynomials.sample_set.ball.center[:,np.newaxis])*rad_ratio + lpolynomials.sample_set.ball.center[:,np.newaxis]

                        best_polynomial = LagrangePolynomials(input_symbols=self.input_symbols, pdegree=2)
                        best_polynomial.initialize(v=new_y, f=None, sort_type=sort_type)  
                        
                    break
            
            if k == max_iter-1:
                print(f"Could not construct polynomials with poisedness < {L} after {max_iter} iterations. Consider increasing the max_iter.") 
        
        return best_polynomial