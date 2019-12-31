import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.integrate import odeint

mu = 8
def vanderpol(X, t):
    x = X[0]
    y = X[1]
    dxdt = y
    dydt = mu*(1-x**2)*y-x
    return [dxdt, dydt]

X0 = [2, 0]
t = np.linspace(0, 2*mu, 700)

sol = odeint(vanderpol, X0, t)
print(sol)

x = sol[:, 0]
y = sol[:, 1]

plt.figure(1)
plt.plot(t,x)
plt.xlabel('Time')
plt.ylabel('Y1 values')
plt.title('Y1 Time graph')
plt.figure(2)
plt.plot(t,y)
plt.xlabel('Time')
plt.ylabel('Y2 values')
plt.title('Y2 Time graph')
plt.show()




print('-------------------------------------------')
def RungeAdaptive( F, tol,a,b,t,y ):
        h = .0001       
        E = tol
        T = np.array([t])
        Y = np.array([y])
        counter = 0
        while t < b:
            if t + h > b:
                h = b - t;
            k1 =  F(t,y)
            k2 =  F(t + h/2.0,y + (h/2)*k1)
            k3 =  F(t + h/2.0,y + (h/2)*k2  )
            z3 =  F(t, y - h*k1 + 2*h*k2)
            k4 =  F(t + h, y+h*k3)  
            Eold = E  
            E = LA.norm(2*k2 + z3 - 2*k3 - k4)*(h/6)
            if len(np.shape(E)) > 0:
                E = max (E)
            if E <= tol:
                t += h
                y += (h/6)*(k1 + 2*k2 + 2*k3 +k4)
                T = np.append(T, t )
                Y = np.append(Y, [y], 0)
            h = h * ( tol/E)**(2/12) * ( tol/Eold)**(-1/12)
            counter = counter + 1
            print(counter)
        plt.figure(3)    
        plt.plot(T, Y[:,0])
        plt.xlabel('Time')
        plt.ylabel('Y1 Values')
        plt.title('Y1 Time Graph')
        plt.figure(4)
        plt.plot(Y[:,0],Y[:,1])
        plt.xlabel('Y1 Values')
        plt.ylabel('Y2 Values')
        plt.title('Phase Portrait')
        plt.figure(5)
        plt.plot(T,Y[:,1])
        plt.xlabel('Time')
        plt.ylabel('Y2 Values')
        plt.title('Y2 Time Graph')
        plt.show()
        return T    
        return Y
        

mu =  10   
def F(t,y):
    F = np.zeros(2)
    # Ley y0 = y and y1 = y'
    F[0] = y[1]               
    F[1] = mu*(1-y[0]**2)*y[1] - y[0]      
    return F
y = np.array([2.0, 0.0])     
Solutions = RungeAdaptive( F, .000001,0,2*mu,0,y )
print(Solutions)



