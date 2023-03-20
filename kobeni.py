import numpy as np
import matplotlib.pyplot as plt


class Linear_reg:
        
        def __init__(self,w,b):
                self.__w = w
                self.__b = b
                self.__J_history = []
        def reset_para(self,w,b):
                self.__w = w
                self.__b = b

        def parameters(self):
                return self.__w,self.__b
        def cost_history(self):
                return self.__J_history                
        #comuting gradient descent---------------------------------------------------------------------        
        def compute_gd(self,x,y,num_iter,alpha,rate,c=False,p=False,d=False):
        #rate of printing
        #for example rate = 100 print every 100 iteration
        #track
        #you want to track what
        # Th following variables indicate whch elements you want to print
        # c: cost function
        # d: the derivatives
        # p w,b
               
                
                for i in range(num_iter):
                        dj_dw, dj_db = self.__derivatives(x,y)
                        self.__w = self.__w - alpha*dj_dw
                        self.__b = self.__b - alpha*dj_db
                        self.__J_history.append(self.__compute_cost(x,y))
                        if(i%rate == 0 and (c or p or d)):
                                # s = f"{i} : cost = {compute_cost(x,y,w,b,f)},dj_dw = {dj_dw}, dj_db = {dj_db}, w = {w}, b = {b:0.2f}"
                                s = f"{i} : "
                                if c:
                                        s = f"cost = {self.__compute_cost(x,y)}, "
                                if p:
                                        s += f"w = {self.__w}, b = {self.__b:0.2f}, "
                                if d:              
                                        s += f"dj_dw = {dj_dw}, dj_db = {dj_db}"
                                
                                print(s)                
                
                
                print(f"w,b found by gradient descent: w: {self.__w}, b: {self.__b:0.2f}")   
                
                
        #predict
        def predict(self,x):
                return self.__compute_fwb(x)
        #calculating fwb--------------------------------------------------------------------------------
        def __compute_fwb(self,x):
                m = x.shape[0]
               
                f_wb = np.zeros(m)

                for i in range(m):
                        f_wb[i] = np.dot(self.__w,x[i])+self.__b
                return f_wb
        """calculating the derivatives------------------------------------------------------------------
        dj_dw now is a vector because
        we need to calculate the derivatives
        for every wi because w is also a vector"""          
        def __derivatives(self,x,y):
                
                m = x.shape[0]
                n = x.shape[1]

                f_wb = self.__compute_fwb(x)
                dj_dw = np.zeros(n)

                dj_db=0.0
                for i in range(m):
                        for j in range(n):
                                dj_dw[j] += ((f_wb[i]-y[i])*x[i][j])/m
                        dj_db += ((f_wb[i]-y[i]))/m
                return dj_dw, dj_db    
        #Calculating the cost functio-----------------------------------------------------------------
        def __compute_cost(self,x, y):
                
                fwb = self.__compute_fwb(x)
                m = x.shape[0]
                cost = 0.0
                for i in range(m):
                        p = pow(fwb[i]-y[i],2)
                        cost += 0.5/m*p
                
                return cost  
        
                  


#--------------------------------------------------------------------------------------------------------



            
class Logistic_reg:
        
        def __init__(self,w,b):
                self.__w = w
                self.__b = b
                self.__J_history = []
        def reset_para(self,w,b):
                self.__w = w
                self.__b = b

        def parameters(self):
                return self.__w,self.__b
        def cost_history(self):
                return self.__J_history                
        #comuting gradient descent---------------------------------------------------------------------        
        def compute_gd(self,x,y,num_iter,alpha,rate,c=False,p=False,d=False):
        #rate of printing
        #for example rate = 100 print every 100 iteration
        #track
        #you want to track what
        # Th following variables indicate whch elements you want to print
        # c: cost function
        # d: the derivatives
        # p w,b
               
                
                for i in range(num_iter):
                        dj_dw, dj_db = self.__derivatives(x,y)
                        self.__w = self.__w - alpha*dj_dw
                        self.__b = self.__b - alpha*dj_db
                        self.__J_history.append(self.__compute_cost(x,y))
                        if(i%rate == 0 and (c or p or d)):
                                # s = f"{i} : cost = {compute_cost(x,y,w,b,f)},dj_dw = {dj_dw}, dj_db = {dj_db}, w = {w}, b = {b:0.2f}"
                                s = f"{i} : "
                                if c:
                                        s = f"cost = {self.__compute_cost(x,y)}, "
                                if p:
                                        s += f"w = {self.__w}, b = {self.__b:0.2f}, "
                                if d:              
                                        s += f"dj_dw = {dj_dw}, dj_db = {dj_db}"
                                
                                print(s)                
                
                
                print(f"w,b found by gradient descent: w: {self.__w}, b: {self.__b:0.2f}")   
                
                
        #predict
        def predict(self,x):
                return self.__compute_fwb(x)
        #calculating fwb--------------------------------------------------------------------------------
        def __sigmoid(self,x) :
                fwb = np.dot(x,self.__w)+self.__b
                s = 1/(1+np.exp(-fwb))
                return s
        """calculating the derivatives------------------------------------------------------------------
        dj_dw now is a vector because
        we need to calculate the derivatives
        for every wi because w is also a vector"""          
        def __derivatives(self,x,y):
                
                m = x.shape[0]
                n = x.shape[1]

                f_wb = self.__sigmoid(x)
                dj_dw = np.zeros(n)

                dj_db=0.0
                for i in range(m):
                        for j in range(n):
                                dj_dw[j] += ((f_wb[i]-y[i])*x[i][j])/m
                        dj_db += ((f_wb[i]-y[i]))/m
                return dj_dw, dj_db    
        #Calculating the cost functio-----------------------------------------------------------------
        def __compute_cost(self,x, y):
                
                fwb = np.dot(x,self.__w)+self.__b
                m = x.shape[0]
                cost = 0.0
                for i in range(m):
                        
                        cost +=    -(1/m)*y[i]*np.log(fwb[i])+(1-y[i])*np.log(1-fwb[i])
                
                return cost  

class Scaler:
        def __init__():
                # !!!!!!!!!!!!!!!!!!!!!!! mean and stdev are not scalers
                self.stdev = 0.0
                self.mean = 0.0
        def __comp_stdev(x):
                mu = comp_mean(x)
                m = x.shape[0]
                n = x.shape[1]
                sigma = np.zeros(n)
                for j in range(n):
                        for i in range(m):
                        sigma[j] += pow(x[i][j]-mu[j],2)/m
                self.stdev = np.sqrt(sigma)
                return self.stdev
        def __comp_mean(x):
                m = x.shape[0]
                n = x.shape[1]
                mu = np.zeros(n)
                for j in range(n):
                        for i in range(m):
                        mu[j] += x[i][j]/m
                self.mean = mu        
                return self.mean    
                //
        def zscore_norm(x):
                m = x.shape[0]
                n = x.shape[1]
                x_norm = np.zeros(x.shape)
                x_mu = comp_mean(x)
                x_sigma = comp_stdev(x)
                for i in range(m):
                        for j in range(n):
                        x_norm[i][j] = (x[i][j] - x_mu[j])/ x_sigma[j]
                return x_norm,x_mu,x_sigma           

def load_data(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X, y


