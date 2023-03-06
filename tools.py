import numpy as np
import matplotlib.pyplot as plt
""" calculating f_Wb
for every row of x let's call it xi 
where xi is a vector
this function return a vector
"""

def compute_fwb(x,w,b):
        m = x.shape[0]
        f_wb = np.zeros(m)

        for i in range(m):
                f_wb[i] = np.dot(w,x[i])+b
        return f_wb  


"""calculating the derivatives
dj_dw now is a vector because
we need to calculate the derivatives
for every wi because w is also a vector"""

def derivatives(x,y,w_i, b_i,):
        w = w_i
        b = b_i
        m = x.shape[0]
        n = x.shape[1]
       
        f_wb = compute_fwb(x,w,b)
        dj_dw = np.zeros(n)

        dj_db=0.0
        for i in range(m):
           for j in range(n):
                        dj_dw[j] += ((f_wb[i]-y[i])*x[i][j])/m
           dj_db += ((f_wb[i]-y[i]))/m
        return dj_dw, dj_db
# Computing gradient descent
def compute_gd(x,y,w_i,b_i,num_iter,alpha,rate,pon=False):
        #rate of printing
        #for example rate = 100 print every 100 iteration
        #pon print or not
        #to specify if you want to see cost,w,b every rate iteration or not
        w = w_i
        b = b_i
        J_history = []
        for i in range(num_iter):
                dj_dw, dj_db = derivatives(x,y,w,b)
                w = w - alpha*dj_dw
                b = b - alpha*dj_db
                J_history.append(compute_cost(x,y,w,b))
                if(i%rate == 0 and pon):
                        print(f"{i} : cost = {compute_cost(x,y,w,b)}, w = {w}, b = {b:0.2f}")                
        print(f"w,b found by gradient descent: w: {w}, b: {b:0.2f}")        
        return w,b,J_history               
#Computing the cost function
def compute_cost(x, y, w, b):
    fwb = compute_fwb(x, w, b)
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        p = pow(fwb[i]-y[i],2)
        cost += 0.5/m*p
        
    return cost

def load_data(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X, y

def zscore_norm(x):
       mu = np.mean(x,axis=0)
       sigma = np.std(x, axis=0)
       x_norm = (x-mu)/sigma

       return (x_norm, mu, sigma)
