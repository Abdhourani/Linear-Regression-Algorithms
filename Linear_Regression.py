from timeit import default_timer as timer
import numpy as np
import math
import matplotlib.pyplot as plt

# Generate Dataset
def Geenerate_data(n):
    a = np.random.rand()
    b = np.random.rand()
    if n>1000:
        print("The Origin slope (m) is : ", a)
        print("The Origin bias (b) is : ", b)
    X = np.random.rand(n,1)
    Y = a*X + b
    noise = np.random.normal(0,0.05, size = (n , 1))
    Y_noised = Y + noise  
    #plt.scatter(X,Y_noised)
    #plt.show()
    return [X , Y_noised]
        
def Merge(X , x):
    X_merged = np.concatenate((X, x))
    return X_merged


def outliers(n , X , Y):
    xn = np.random.rand(n,1)
    yn = np.random.rand(n,1)
    X = Merge(X , xn)
    Y = Merge(Y , yn) 
    '''
    for i in range(20):
        x , y = Geenerate_data(1)
        X = Merge(X , x)
        Y = Merge(Y , y)
    '''
    return [X , Y]

# 1.Gradiant Descent
def GD(X , Y_noised):
    n = len(X)
    start = timer()
    m = 0.1
    b = 0.1
    L = 0.001
    Dm = 100
    Db = 100
    while (abs(L*Dm) > 1e-8) and (abs(L*Db) > 1e-8):
        #total_err = 0
        for i in range(n):
            Y_pred = m*X + b
            Dm =(-2/n)*np.sum(X*(Y_noised - Y_pred))
            Db =(-2/n)*np.sum(Y_noised - Y_pred)
            m = m - L*Dm
            b = b - L*Db
    Y_pred = m*X + b
    end = timer()
    plt.scatter(X , Y_noised , color = "black")
    plt.scatter(X , Y_pred , color = "red")
    plt.show()
    print("----------------------------------------------")
    print("M = " , m , "B = " , b)
    print("MSE_m  = " , Dm , " MSE_b = " , Db)
    print("Time : " , end - start)
    print("----------------------------------------------")


# 2.Stochastic Gradiant Descent
def SGD(X , Y_noised):
    n = len(X)
    start = timer()
    m = 0.1
    b = 0.1
    L = 0.001
    Dm = 10000
    Db = 10000
    f = 0
    while (abs(L*Dm) > 1e-8) and (abs(L*Db) > 1e-8):
        epoch  = 20
        batch_size = n//20
        for i in range(epoch):
            indexes = np.random.randint(0, len(X), batch_size)
            Xs = np.take(X, indexes)
            Ys = np.take(Y_noised, indexes)
            Ns = len(Xs)
            f = Ys - (m*Xs + b)
            Dm = ((-2/Ns) * Xs.dot(f).sum() )
            Db = ((-2/Ns) * f.sum() )
            m = m - L*Dm
            b = b - L*Db
    Y_pred = m*X + b
    end = timer()
    plt.scatter(X , Y_noised , color = "black")
    plt.scatter(X , Y_pred , color = "red")
    plt.show()
    print("----------------------------------------------")
    print("M = " , m , "B = " , b)
    print("MSE_m  = " , Dm , " MSE_b = " , Db)
    print("Time : " , end - start)
    print("----------------------------------------------")


# 3. The Momentum
def Momentum(X , Y_noised):
    n = len(X)
    start = timer()
    m = 0.1
    b = 0.1
    alpha = 0.001
    btta = 0.9
    Dm = 10000
    Db = 10000
    f = 0
    vm_old = 0
    vb_old = 0
    while (abs(alpha*Dm) > 1e-8) and (abs(alpha*Db) > 1e-8):
        epoch  = 20
        batch_size = n//20
        for i in range(epoch):
            indexes = np.random.randint(0, len(X), batch_size)
            Xs = np.take(X, indexes)
            Ys = np.take(Y_noised, indexes)
            Ns = len(Xs)
            f = Ys - (m*Xs + b)
            Dm = (-2 * Xs.dot(f).sum() / Ns)
            Db = (-2 * f.sum() / Ns)
            vm = btta * vm_old + (1-btta)*Dm
            vb = btta * vb_old + (1-btta)*Db
            m = m - alpha * vm
            b = b - alpha * vb
            vb_old = vb
            vm_old = vm
    Y_pred = m*X + b
    end = timer()
    plt.scatter(X , Y_noised , color = "black")
    plt.scatter(X , Y_pred , color = "red")
    plt.show()
    print("----------------------------------------------")
    print("M = " , m , "B = " , b)
    print("MSE_m  = " , Dm , " MSE_b = " , Db)
    print("Time : " , end - start)
    print("----------------------------------------------")

# 4. Root Mean Square Propagation
def RMS_prop(X , Y_noised):
    n = len(X)
    start = timer()
    m = 0.1
    b = 0.1
    alpha = 0.001
    btta = 0.9
    Dm = 10000
    Db = 10000
    f = 0
    Sm_old = 0
    Sb_old = 0
    eps = 1e-8
    while (abs(alpha*Dm) > 1e-8) and (abs(alpha*Db) > 1e-8):
        epoch  = 20
        batch_size = n/20
        for i in range(epoch):
            indexes = np.random.randint(0, len(X), batch_size)
            Xs = np.take(X, indexes)
            Ys = np.take(Y_noised, indexes)
            Ns = len(Xs)
            f = Ys - (m*Xs + b)
            Dm = (-2/Ns) * Xs.dot(f).sum()
            Db = (-2/Ns) * f.sum()
            Sm = btta * Sm_old + (1-btta)*Dm**2
            Sb = btta * Sb_old + (1-btta)*Db**2
            m = m - (alpha / math.sqrt(Sm + eps) ) * Dm
            b = b - (alpha / math.sqrt(Sb + eps) ) * Db
            Sb_old = Sb
            Sm_old = Sm
    Y_pred = m*X + b
    end = timer()
    plt.scatter(X , Y_noised , color = "black")
    plt.scatter(X , Y_pred , color = "red")
    plt.show()
    print("----------------------------------------------")
    print("M = " , m , "B = " , b)
    print("MSE_m  = " , Dm , " MSE_b = " , Db)
    print("Time : " , end - start)
    print("----------------------------------------------")

# 5. ADAM
def ADAM(X , Y_noised):
    n = len(X)
    start = timer()
    m = 0.1
    b = 0.1
    alpha = 0.001
    btta1 = 0.8
    btta2 = 0.92
    Dm = 10000
    Db = 10000
    f = 0
    Sm_old = 0
    Sb_old = 0
    vm_old = 0
    vb_old = 0
    eps = 1e-8
    k = 1
    while (abs(alpha*Dm) > 1e-8) and (abs(alpha*Db) > 1e-8):
        epoch  = 10
        batch_size = 100
        for i in range(epoch):
            indexes = np.random.randint(0, len(X), batch_size)
            Xs = np.take(X, indexes)
            Ys = np.take(Y_noised, indexes)
            Ns = len(Xs)
            f = Ys - (m*Xs + b)
            Dm = (-2/Ns) * Xs.dot(f).sum()
            Db = (-2/Ns) * f.sum()
            vm = btta1 * vm_old + (1-btta1) * Dm
            Sm = btta2 * Sm_old + (1-btta2) * Dm**2
            vb = btta1 * vb_old + (1-btta1) * Db
            Sb = btta2 * Sb_old + (1-btta2) * Db**2
            vm = vm/(1.0-btta1**(k+1) + eps)
            vb = vb/(1.0-btta1**(k+1) + eps)
            Sm = Sm/(1.0- btta2**(k+1) + eps )
            Sb = Sb/(1.0- btta2**(k+1) +eps)
            m = m - (alpha / math.sqrt(Sm + eps) ) * vm
            b = b - (alpha / math.sqrt(Sb + eps) ) * vb
            Sb_old = Sb
            Sm_old = Sm
        k = k + 1 
    Y_pred = m*X + b
    end = timer()
    plt.scatter(X , Y_noised , color = "black")
    plt.scatter(X , Y_pred , color = "red")
    plt.show()
    print("----------------------------------------------")
    print("M = " , m , "B = " , b)
    print("MSE_m  = " , Dm , " MSE_b = " , Db)
    print("Time : " , end - start)
    print("----------------------------------------------")


X , Y = Geenerate_data(2000)
GD(X , Y)
#SGD(X , Y)
#Momentum(X , Y)     
#RMS_prop(X , Y)
#ADAM(X , Y)


#X , Y = outliers(20 , X , Y)
#GD(X , Y)
#SGD(X , Y)
#Momentum(X , Y)     
#RMS_prop(X , Y)
#ADAM(X , Y)





