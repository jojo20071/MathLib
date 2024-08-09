import numpy as np

def add(a,b):
    return a+b

def lerp(a,b,t):
    return a*(1-t)+b*(t) 
    

def linear_bezier(a,b,t):
    return np.array(lerp(a,b,t))

def quadratic_bezier(a,b,c,t):
    return np.array((1-t)**2*a+2*(1-t)*t*b+t**2*c)

def cubic_bezier(a,b,c,d,t):
    return np.array((1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t**2*c+t**3*d)





