"""
This problem is taken from Exercise 1.9 of Charles Audet book: Derivative-free and Blakcbox Optimization

Model equation is (with symbol simplification): 

a(x) = a_0 (1 + c^2 x^2)^(b-1)/2

with error function

e_i(a_0, c, b) = [a_0 (1 + c^2 x^2)^(b-1)/2 - a_i]

smooth function: 
gnonsmooth = sum e_i
gsmooth = sum e^2

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(df:pd.DataFrame):

    # Exercise 1.9 a) 
    errorsuma = error_function(5200, 140, 0.38, df, sumtype="Nonsmooth")
    print(f'Non smooth function g(5200, 140, 0.38) = {errorsuma}')
    errorsuma = error_function(5252.0, 141.4, 0.3838, df, sumtype="Nonsmooth")
    print(f'Non smooth function g(5252.0, 141.4, 0.3838) = {errorsuma}')

    # Exercise 1.9 b) 
    errorsumb = error_function(5200, 140, 0.38, df, sumtype="Smooth")
    print(f'Smooth function g(5200, 140, 0.38) = {errorsumb}')
    errorsumb = error_function(5252.0, 141.4, 0.3838, df, sumtype="Smooth")
    print(f'Smooth function g(5252.0, 141.4, 0.3838) = {errorsumb}')

    # Exercise 1.9 c)
    predicted_viscosity = []
    for i in range(df.shape[0]):
        visc = viscosity_eval(df['Strain rate'].iloc[i], 5200, 140, 0.38)
        predicted_viscosity.append(visc)
    df['Pred Viscosity'] = predicted_viscosity

    fig, ax = plt.subplots()
    ax.plot(df['Strain rate'], df['Viscosity'], label='Original data')
    ax.plot(df['Strain rate'], df['Pred Viscosity'], label='Predicted data')
    plt.xlabel('Strain rate $\dot{\gamma}_i (s^{-1})$')
    plt.ylabel('Viscosity $ \eta (Pa . s)$')
    plt.yscale("log")
    plt.legend()
    filename = './Audet_Ch1_Ex1_9_c.png'
    plt.savefig(filename)
    
    return

def error_function(a_0:float, c:float, b:float, df:pd.DataFrame, sumtype:str='Nonsmooth') -> float:

    Ndata = df.shape[0]

    errorsum = 0
    for i in range(Ndata):

        error = error_eval(df['Strain rate'].iloc[i], df['Viscosity'].iloc[i], a_0, c, b)

        if sumtype == "Nonsmooth":
            errorsum += error
        elif sumtype == "Smooth":
            errorsum += error**2
        else:
            raise Exception(f"sumtype must be 'Nonsmooth' or 'Smooth'. Got {sumtype}")

    return errorsum

def error_eval(x:float, a_i:float, a_0:float, c:float, b:float) -> float:
    return np.abs(viscosity_eval(x, a_0, c, b) - a_i)

def viscosity_eval(x:float, a_0:float, c:float, b:float) -> float:
    return a_0 * (1 + c**2 * x**2)**((b-1)/2)

if __name__ == '__main__':

    df = pd.DataFrame()

    # Strain rate
    df['Strain rate'] = [0.0137, 0.0274, 0.0434, 0.0866, 0.137, 0.274, 0.434, 0.866, 1.37, 2.74, 4.34, 5.46, 6.88]
    # Viscosity
    df['Viscosity'] = [3220, 2190, 1640, 1050, 766, 490, 348, 223, 163, 104, 76.7, 68.1, 58.2]

    main(df)

