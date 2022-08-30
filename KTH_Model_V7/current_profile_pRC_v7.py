import numpy as np
import matplotlib.pyplot as plt


#is called by tail field neutralsheet

def current_profile_pRC_v7(rho, control_params):
    #this function calculates the current profile of the neutralcurrentsheet
    
    d = control_params['d']
    e = control_params['e']
    f = control_params['f']
    
    
    current = -(d/f*np.sqrt(2*np.pi)) * np.exp(-(rho-e)**2/(2*f**2))
    
    #plt.figure()
    #plt.plot(-rho, current)
    #plt.title('Current from current_profile_pRC.py')
    
    return current