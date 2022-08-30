import numpy as np
import matplotlib.pyplot as plt


#is called by tail field neutralsheet

def current_profile_v7(rho, control_params):
    #this function calculates the current profile of the neutralcurrentsheet
    
    a = control_params['a']
    b = 1.0
    c = control_params['c']
    
    
    current = a * (-rho + b) **2 * np.exp(-c * (-rho+b)**2)
    
    #plt.figure()
    #plt.plot(-rho, current)
    #plt.title('Current from current_profile.py')
    
    return current