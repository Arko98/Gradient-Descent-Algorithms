#Helper Functions
def f(x,w,b):
    '''Sigmoid Function'''
    f = 1/(1+np.exp(-(w*x+b)))
    return f
def mse(x,y,w,b):
    '''Mean Squared Loss Function'''
    L = 0.0
    for i in range(x.shape[0]):
        L += 0.5*(y[i]-f(x[i],w,b))**2
    return L
def cross_entropy(x,y,w,b):
    '''Cross Entropy Loss Function'''
    L = 0.0
    for i in range(x.shape[0]):
        L += -(y[i]*np.log(f(x[i],w,b)))
    return L
def grad_w_mse(x,y,w,b):
    fx = f(x,w,b) 
    dw = (fx - y)*fx*(1-fx)*x
    return dw
def grad_b_mse(x,y,w,b):
    fx = f(x,w,b) 
    db = (fx - y)*fx*(1-fx)
    return db
def grad_w_cross(x,y,w,b):
    fx = f(x,w,b) 
    dw = (- y)*(1-fx)*x
    return dw
def grad_b_cross(x,y,w,b):
    fx = f(x,w,b) 
    db = (- y)*(1-fx)
    return db

#Algorithm
#Gradient Discent
def Decay_GD(x,y,epochs,batch_size,loss,lr):
    w = np.random.randn()
    b = np.random.randn()
    count = 0                      #A counter to see how many times the iteration ran
    l_list = []
    w_list = []
    b_list = []
    w_cache,b_cache = w,b          #Caches for current epoch
    w_prev,b_prev = 0,0            #Contain previous parameters
    points,epoch_val = 0,0
    prev_loss,current_loss = 100,0 #High Value for first previous loss
    ep = [i for i in range(epochs+1)]
    dw,db = 0,0
    while (epoch_val <= epochs):
        count += 1
        dw,db = 0,0
        w_prev, b_prev = w_cache,b_cache
        for j in range(x.shape[0]):
            if (loss == 'mse'):
                dw += grad_w_mse(x[j],y[j],w_cache,b_cache)
                db += grad_b_mse(x[j],y[j],w_cache,b_cache)
            elif (loss == 'cross_entropy'):
                dw += grad_w_cross(x[j],y[j],w_cache,b_cache)
                db += grad_b_cross(x[j],y[j],w_cache,b_cache)
            points += 1
            if(points % batch_size == 0):
                w_cache = w_cache - lr*dw
                b_cache = b_cache - lr*db
                dw,db = 0,0
        if (loss == 'mse'):     
            current_loss = mse(x,y,w_cache,b_cache)[0]
        elif (loss == 'cross_entropy'):
            current_loss = cross_entropy(x,y,w_cache,b_cache)[0]
        #Successful Epoch
        if (current_loss < prev_loss):
            epoch_val += 1              
            prev_loss = current_loss
            #Load the new updates of parameters
            print('Loss after {}th epoch = {}\n'.format(epoch_val,current_loss))
            l_list.append(current_loss)
            w_list.append(w_cache[0])
            b_list.append(b_cache[0])
        elif (current_loss >= prev_loss):
            lr = lr/2
            w_cache,b_cache = w_prev,b_prev
    print('\n\nDecaying Learning rate Gradient ran for {} iterations for {} epochs\n\n'.format(count,epochs))        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch Curve\nAlgotithm :Mini Batch Decaying Learning Rate Gradient Decent\nBatch Size = {}\nInitial Learning Rate = {}\nLoss Function = {}'.format(batch_size,lr,loss))
    plt.plot(ep,l_list)
    plt.show()
    return w_list,b_list