class IncorrectConstantsException(Exception):
    """Raised when constants requirement are not met"""
    pass

class TRQPIncompatible(Exception):
    """Raised when TRQP is NOT Compatible"""
    pass

class EndOfAlgorithm(Exception):
    """
    Raised when : 
    
    1. restoration phase is impossible to compute
    """
    pass

class PoisednessIsZeroException(Exception):
    "Raise when, yeah, poisedness is zero. Usually because of duplicated points"
    pass

class SolutionFound(Exception):
    pass