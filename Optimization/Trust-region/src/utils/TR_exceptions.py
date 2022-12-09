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