import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def P(x,y): 
    a = 3
    b = 9
    c = 15
    d = 15
    xprime = a *x - b*x*y
    yprime = c*x*y - d*y  
    return xprime, yprime  
               
def RungeAdaptive( F, tol, hmax, hmin,a,b ):
        t = 0
        x = 1#IC
        y = 1
        h = hmax  #will be endpoint - satrt point / steps wanted      
        T = np.array([t])
        Y = np.array([y])
        X = np.array([x])
        while t < b:
            if t + h > b:
                h = b - t;
            k1, y1 =  F(x,y)
            k2, y2 =  F(x + (h/2)*k1,y + (h/2)*y1)
            k3, y3 =  F(x + (h/2)*k2,y + (h/2)*y2  )
            z3, Z3 =  F( x - h*k1 + 2*h*k2, y - h*y1 + 2*h*y2)
            k4, y4 =  F(x + h*k3, y+h*y3)    
            rx = LA.norm(2*k2 + z3 - 2*k3 - k4)*(h/6)
            R = LA.norm(2*y2 + Z3 - 2*y3 - y4)*(h/6)
            r = LA.norm([rx, R])
            if len(np.shape(r)) > 0:
                r = max (r)
            if r <= tol:
                t += h
                x += (h/6)*(k1 + 2*k2 + 2*k3 +k4)
                y += (h/6)*(y1 + 2*y2 + 2*y3 +y4)
                T = np.append(T, t )
                Y = np.append(Y, [y], 0)
                X = np.append(X, [x], 0 )
            h = h * ( tol/r)**0.25
            if h > hmax:
                h = hmax
            elif h < hmin:
                break
            plt.figure(1)
            plt.plot(X, Y)
            plt.title('Phase Portrait')
            plt.figure(2)
            plt.plot(T, X)
            plt.plot(T, Y)
            plt.xlabel('Time values')
            plt.ylabel('X & Y Values')
            plt.title('Predator-Prey LogLog')
            plt.show()
        return (T, X , Y) #T is for time and X is the actual Approximation
                    
A = RungeAdaptive(P, .000001, .1, .0001, 0, 10)   
print(A)             

