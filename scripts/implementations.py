import numpy as np
from given_costs import *
from given_proj1_helpers import *




def compute_gradient(y, tx, w):
    """ Compute the gradient.  """
    e = y - np.dot(tx,w)
    gradL = -1/len(y) * (np.dot(np.transpose(tx),e))
    
    return gradL


def least_squares(y, tx):
    """Least squares regression using normal equations"""
    ttx = np.transpose(tx)
    A = np.dot(ttx,tx)
    b = np.dot(ttx,y)
    w = np.linalg.solve(A,b)
    loss = compute_loss(y, tx, w)
    return w, loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
        
    w = initial_w
    
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        
        w = w - gamma * gradient
        loss = compute_loss(y, tx, w)
        
        # important use an odd number...to see jumps
        #if (n_iter % 11 == 0):
        #   print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #        bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss



def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    batch_size = 250
        
    w = initial_w
        
    for n_iter in range(max_iters):
        
        j = 0
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=5):
            j+=1
            gradient = compute_gradient(minibatch_y, minibatch_tx, w) 
            
            # update w by gradient
            w = w - gamma * gradient
            loss = compute_loss(y, tx, w)
            
        #if (n_iter % 101 == 0):
        #    print("Gradient Descent(epoch :{bi}/{ti}): minibatch={m} loss={l}, w0={w0}, w1={w1}".format(
        #        bi=n_iter, ti=max_iters - 1, m=j, l=loss, w0=w[0], w1=w[1]))   
            
    return w, loss


def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations """
    txt = np.transpose(tx)
    
    XTXL = np.dot(txt,tx) + lambda_*np.identity(tx.shape[1])    
  
    w = np.dot(np.dot(np.linalg.inv(XTXL),txt),y)
    
    return w, compute_loss(y, tx, w)




def sigmoid(t):
    """apply sigmoid function on t."""
    t[t>745]=745
    t[t<-709]=-709
    return 1/(1+np.exp(-t))

def compute_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.dot(np.transpose(tx),sigmoid(np.dot(tx,w))) - np.c_[np.dot(np.transpose(tx),y)]

def compute_logistic_hessian(y, tx, w):
    """return the hessian of the loss function."""
    sigma = sigmoid(np.dot(tx,w))
    diag_S = sigma*(1-sigma)
    return np.dot(np.dot(np.transpose(tx),np.diag(np.ravel(diag_S))),tx)

def logistic_regression(y, tx, gamma, max_iters):
    """Logistic regression using gradient descent or SGD"""
    
    threshold = 1e-8
    losses = []
    mse_losses = []
    beta = 0.1
    
    # initialisation
    w = np.zeros((tx.shape[1], 1))
    mse_losses.append(compute_loss(y, tx, np.ndarray.flatten(w)))
    losses.append(compute_logistic_loss(y, tx, w))
    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        gradient = compute_logistic_gradient(y, tx, w)
        w_next = w - gamma*gradient
        loss_next = compute_logistic_loss(y, tx, w_next)
        
        # update step size criteria
        
        mse_loss_next = compute_loss(y, tx, np.ndarray.flatten(w_next))
        
        if mse_loss_next > mse_losses[-1]:
            gamma = beta*gamma
        else:
            w = w_next
            losses.append(loss_next)
            mse_losses.append(mse_loss_next)
        
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=losses[-1]))
            print("The loss={l}".format(l=mse_losses[-1]))
            print("gamma={g}".format(g=gamma))
        
        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print(iter)
            break
    # visualization
    loss = compute_logistic_loss(y, tx, w)
    print("The loss={l}".format(l=loss))
    print("The mse loss={l}".format(l=compute_loss(y, tx, np.ndarray.flatten(w))))

    return w,loss


def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
    """ Regularized logistic regression using gradient descent or SGD"""
    threshold = 1e-8
    losses = []
    mse_losses = []
    beta = 0.1
    
    # initialisation
    w = np.zeros((tx.shape[1], 1))
    mse_losses.append(compute_loss(y, tx, np.ndarray.flatten(w)))
    losses.append(compute_logistic_loss(y, tx, w) + lambda_*np.sum(w*w))

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        gradient = compute_logistic_gradient(y, tx, w) + 2*lambda_*w
        w_next = w - gamma*gradient
        loss_next = compute_logistic_loss(y, tx, w_next) + lambda_*np.sum(w_next*w_next)
        
        # update step size criteria
        
        mse_loss_next = compute_loss(y, tx, np.ndarray.flatten(w_next))
        
        if mse_loss_next > mse_losses[-1]:
            gamma = beta*gamma
        else:
            w = w_next
            losses.append(loss_next)
            mse_losses.append(mse_loss_next)
        
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=losses[-1]))
            print("The loss={l}".format(l=mse_losses[-1]))
            print("gamma={g}".format(g=gamma))
        
        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print(iter)
            break
    # visualization
    loss = compute_logistic_loss(y, tx, w)
    print("The loss={l}".format(l=loss))
    print("The mse loss={l}".format(l=compute_loss(y, tx, np.ndarray.flatten(w))))

    return w,loss

def logistic_regression_newton(y, tx, gamma, max_iters):
    """ """
    threshold = 1e-8
    losses = []
    mse_losses = []
    beta = 0.1
    
    # initialisation
    w = np.zeros((tx.shape[1], 1))
    mse_losses.append(compute_loss(y, tx, np.ndarray.flatten(w)))
    losses.append(compute_logistic_loss(y, tx, w))
    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        gradient = compute_logistic_gradient(y, tx, w)
        hessian = compute_logistic_hessian(y, tx, w)
        w_next = w - gamma*np.linalg.solve(hessian, gradient)
        loss_next = compute_logistic_loss(y, tx, w_next)
        
        # update step size criteria
        
        mse_loss_next = compute_loss(y, tx, np.ndarray.flatten(w_next))
        
        if mse_loss_next > mse_losses[-1]:
            gamma = beta*gamma
        else:
            w = w_next
            losses.append(loss_next)
            mse_losses.append(mse_loss_next)
            
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=losses[-1]))
            print("The loss={l}".format(l=mse_losses[-1]))
            print("gamma={g}".format(g=gamma))
        
        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print(iter)
            break
    # visualization
    loss = compute_logistic_loss(y, tx, w)
    print("The loss={l}".format(l=loss))
    print("The mse loss={l}".format(l=compute_loss(y, tx, np.ndarray.flatten(w))))

    return w,loss


def reg_logistic_regression_newton(y, tx, lambda_, gamma, max_iters):
    """Regularized logistic regression using gradient descent or SGD"""
    threshold = 1e-8
    losses = []
    mse_losses = []
    beta = 0.1
    
    # initialisation
    w = np.zeros((tx.shape[1], 1))
    mse_losses.append(compute_loss(y, tx, np.ndarray.flatten(w)))
    losses.append(compute_logistic_loss(y, tx, w) + lambda_*np.sum(w*w))
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        gradient = compute_logistic_gradient(y, tx, w) + 2*lambda_*w
        hessian = compute_logistic_hessian(y, tx, w) + 2*lambda_*np.eye(w.shape[0])
        w_next = w - gamma*np.linalg.solve(hessian, gradient)
        loss_next = compute_logistic_loss(y, tx, w_next) + lambda_*np.sum(w_next*w_next)

        # update step size criteria
        
        mse_loss_next = compute_loss(y, tx, np.ndarray.flatten(w_next))
        
        if mse_loss_next > mse_losses[-1]:
            gamma = beta*gamma
        else:
            w = w_next
            losses.append(loss_next)
            mse_losses.append(mse_loss_next)
        
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=losses[-1]))
            print("The loss={l}".format(l=mse_losses[-1]))
            print("gamma={g}".format(g=gamma))
        
        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print(iter)
            break
    # visualization
    loss = compute_logistic_loss(y, tx, w)
    print("The loss={l}".format(l=loss))
    print("The mse loss={l}".format(l=compute_loss(y, tx, np.ndarray.flatten(w))))

    return w,loss


def lasso(y, tx, lambda_, max_iters=20):
    """ Lasso Regression - doesnt work well, need to be improved"""    
    
    ##initialize the Lasso solution
    txt = np.transpose(tx)
    
    #This assumes that the penalty is lambda * w'*w instead of lambda * ||w||_1
    w = np.linalg.solve((np.dot(txt, tx) + 2*lambda_), np.dot(txt,y));
    
    ##start while loop

    #convergence flag
    found = False;
    
    #convergence tolerance
    TOL = 1e-6;
    
    iters = 0
    
    while not found and iters < max_iters:
    
        #save current w
        w_old = np.copy(w);
        
        #optimize elements of w one by one
        for i in range(tx.shape[1]) :
            
            #optimize element i of w
            
            #get ith col of X
            xi = tx[:, i];
            
            #get residual excluding ith col
            yi = (y - np.dot(tx, w)) + xi*w[i];           
            
            #calulate xi'*yi and see where it falls
            deltai = np.dot(np.transpose(xi),yi) #1 by 1 scalar
            
            if deltai < -lambda_ :
                
                w[i] = ( deltai + lambda_ )/ np.dot(np.transpose(xi),xi);
            
            elif deltai > lambda_ :
            
                w[i] = ( deltai - lambda_ )/np.dot(np.transpose(xi),xi);
            
            else :
                w[i] = 0;
            
        iters += 1
        if iters % 10 == 0:
            print("iter : ", iters, " loss = ", compute_loss(y, tx, w))
        
        #check difference between w and w_old
        if np.amax(np.absolute(w - w_old)) <= TOL :
            found = True;
        
    return w, compute_loss(y, tx, w)