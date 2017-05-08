import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from matplotlib.animation import FuncAnimation

def gaussian(mean,var, x_min, x_max, N):
    x = np.linspace(x_min,x_max,N)
    return x, np.exp(-(x - mean)**2/(2*var))/(2*var*3.14150)**(0.5)

def evaluate_gaussian(x, mean,var):
    return np.exp(-(x - mean)**2/(2*var))/(2*var*3.14150)**(0.5)

def plot_likelihood(X, mu_est, var_est, mu_range = [-0.5,0.5], var_range = [0.5, 2], N = 101):
    variances = np.linspace(*var_range, N)
    mus = np.linspace(*mu_range,N)
    likelihood = np.zeros((N,N))
    for imu, mu in enumerate(mus):
        for ivar, var in enumerate(variances):
            # Si no uso el logaritmo, todos los valores del producto son cero debido a problemas numericos
            likelihood[imu,ivar] = np.sum(np.log(evaluate_gaussian(X,mu,var)))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Xp, Yp = np.meshgrid(mus, variances)
    ax.scatter(mu_est,var_est, np.sum(np.log(evaluate_gaussian(X,mu_est,var_est))), color='y', s = 100)
    ax.plot_surface(Xp, Yp, likelihood.T, cmap=cm.coolwarm,linewidth=0, antialiased=True)
    ax.set_xlabel('medias')
    ax.set_ylabel('varianzas')
    return mus, variances, likelihood

def get_linear_set(A = 4, B = 2, mean = 0, sigma = 6, N = 40, ratio = 0.25):
    Div = int(N*ratio)
    X_total = np.linspace(0,N-1, N).reshape(N,1)
    np.random.shuffle(X_total)

    X_LS = X_total[:Div]
    Y_LS = A + B*X_LS + np.random.normal(mean,sigma, len(X_LS)).reshape(len(X_LS),1)

    X_LS_test = X_total[Div:]
    Y_LS_test = A + B*X_LS_test + np.random.normal(mean,sigma, len(X_LS_test)).reshape(len(X_LS_test),1)
    return X_LS, Y_LS, X_LS_test, Y_LS_test

def get_polynomial_set(coefs, mean = 0, sigma = 6, N = 40, ratio = 0.25):
    Div = int(N*ratio)
    X_total = np.linspace(0,1, N).reshape(N,1)
    np.random.shuffle(X_total)

    X_LS = X_total[:Div]
    X_LS_test = X_total[Div:]
    Y_LS = 0 
    Y_LS_test = 0
    for i, coef in enumerate(coefs):
        Y_LS = Y_LS + coef*(X_LS**i)
        Y_LS_test = Y_LS_test + coef*(X_LS_test**i)

    Y_LS = Y_LS + np.random.normal(mean,sigma, len(X_LS)).reshape(len(X_LS),1)
    Y_LS_test = Y_LS_test + np.random.normal(mean,sigma, len(X_LS_test)).reshape(len(X_LS_test),1)
    return X_LS, Y_LS, X_LS_test, Y_LS_test

def get_matrix_polinomial_model(X,Y, N = 1, normalize = False):
    X_all = np.zeros((X.shape[0], X.shape[1]+N))
    X_ones = np.ones((X.shape[0], 1)).flatten()
    X_mat = np.zeros((X.shape[0], X.shape[1]+N-1))
    means = np.zeros((X.shape[1]+N,1)).flatten()
    stds = np.ones((X.shape[1]+N,1)).flatten()
    for i in range(N):
        X_mat[:,i:] = X**(i+1)
    
    X_all[:,0] = X_ones
    X_all[:,1:] = X_mat
    
    if normalize:
        means[1:] = X_mat.mean(axis=0)
        stds[1:] = X_mat.std(axis=0)
        X_all = (X_all - means)/stds
    return X_all, means, stds

def get_parameters_polinomial_model_lasso(X,Y, lamb = 0, N = 1, normalize = False, fit_intercept=True, max_iter=100000):
    X, means, stds = get_matrix_polinomial_model(X,Y, N = N, normalize = normalize)
    clf = linear_model.Lasso(alpha=lamb, normalize= normalize, fit_intercept=fit_intercept, max_iter=100000)
    clf.fit(X, Y)
    coefs = clf.coef_
    if coefs.shape == ():
        coefs = coefs.reshape(1,1)
    return coefs, clf.predict(X) , means, stds

def get_parameters_polinomial_model_linear(X,Y, N = 1, normalize = False, fit_intercept=False, lamb = 0):
    X, means, stds = get_matrix_polinomial_model(X,Y, N = N, normalize = normalize)
    clf = linear_model.Ridge(alpha = lamb, normalize= normalize, fit_intercept=fit_intercept)
    if (fit_intercept):
        clf.fit(X[:,1:], Y)
    else:
        clf.fit(X, Y)
    coefs = clf.coef_
    if coefs.shape == ():
        coefs = coefs.reshape(1,1)
    return coefs, clf.predict(X) , means, stds

def get_parameters_polinomial_model(X,Y, lamb = 0, N = 1, normalize = False, fit_intercept=True):
    X_all, means, stds = get_matrix_polinomial_model(X,Y, N = N, normalize = normalize)
    
    lamb_mat = lamb * np.eye(X.shape[1]+N)
    lamb_mat[0,0] = 0
    theta = np.linalg.inv(lamb * np.eye(X.shape[1]+N) + np.dot(X_all.T,X_all)).dot(X_all.T).dot(Y)
    #return theta, X_all.dot(theta) + lamb*theta[1:].T.dot(theta[1:]).flatten(), means, stds
    return theta, X_all.dot(theta) , means, stds

def get_plot_polinomial_estimations(X_LS, Y_LS, X_LS_test, Y_LS_test, model = get_parameters_polinomial_model, N = 40, orders = [0, 1, 3, 4, 5, 8], lamb = 0, normalize = False, max_order = 9, fit_intercept=False, plot_test = True, figsize=(12, 10)):
    if len(orders)>1:
        f, (ax_arr) = plt.subplots(int(len(orders)/2),2, sharex=False, sharey=False, figsize=figsize)
        ax_arr = ax_arr.flatten()
    else:
        f, (ax_arr) = plt.subplots(1,1, sharex=False, sharey=False, figsize=figsize)
        ax_arr = [ax_arr]
    x_lin = np.linspace(min([min(X_LS),min(X_LS_test)]),max(X_LS),1000)
    MSEs = []
    MSEs_test = []
    plt_num = 0
    for i in range(max_order+1):
        #thetas, points, means, stds  = get_parameters_polinomial_model_lasso(X_LS,Y_LS, lamb = lamb, N = i, normalize = normalize, fit_intercept=True)
        thetas, points, means, stds = model(X_LS,Y_LS, N = i, lamb = lamb, normalize = normalize, fit_intercept=fit_intercept)
        MSEs.append(np.sum((points - Y_LS)**2)/len(points))
        y_est = 0
        Y_LS_test_est = 0
        for j, theta in enumerate(thetas.flatten()):
            x_lin_norm = (x_lin**j-means[j])/stds[j] 
            y_est = y_est + theta*(x_lin_norm)
            X_LS_test_norm = (X_LS_test**j-means[j])/stds[j]
            Y_LS_test_est = Y_LS_test_est + theta*(X_LS_test_norm) 
        #y_est = y_est + lamb*thetas[1:].T.dot(thetas[1:]).flatten()
        #Y_LS_test_est = Y_LS_test_est + lamb*thetas[1:].T.dot(thetas[1:]).flatten()

        MSEs_test.append(np.sum((Y_LS_test_est - Y_LS_test)**2)/len(Y_LS_test))
        if (i in orders):
            #ax_arr[plt_num].scatter(X_LS, points, color = 'k')
            ax_arr[plt_num].scatter(X_LS, Y_LS, color = 'b')
            ax_arr[plt_num].plot(x_lin, y_est)
            ax_arr[plt_num].set_xlim([0,max(X_LS)])
            if plot_test:
                ax_arr[plt_num].scatter(X_LS_test, Y_LS_test, color = 'y')
                ax_arr[plt_num].set_ylim([min(min(Y_LS), min(Y_LS_test)),max(max(Y_LS), max(Y_LS_test))])
            else:
                ax_arr[plt_num].set_ylim([min(min(Y_LS)),max(max(Y_LS))])
            poly_str = 'w_0'
            for u in range(i):
                poly_str = poly_str + '+w_{'+str(u+1)+'}x^{'+str(u+1)+'}'
            ax_arr[plt_num].set_title(r'$'+poly_str+'$')      
            plt_num = plt_num + 1      
    return np.array(MSEs), np.array(MSEs_test)

def get_loss_polys(X_LS,Y_LS, model, max_order, lamb = 0, normalize = True, fit_intercept=False):
    J_array = []
    for i in range(max_order+1):
        _, points, _, _ = model(X_LS,Y_LS, N = i, lamb = lamb, normalize = normalize, fit_intercept=fit_intercept)
        J = np.sum((points - Y_LS)**2)/(2*len(points))
        J_array.append(J)
    return J_array

def animate_poly(i, ax, line, model, x_lin, X_LS,Y_LS, lamb , normalize, fit_intercept):
    thetas, points, means, stds = model(X_LS,Y_LS, N = i, lamb = lamb, normalize = normalize, fit_intercept=fit_intercept)
    J = 10*np.log(np.sum((points - Y_LS)**2)/(2*len(points)))
    y_est = 0
    Y_LS_test_est = 0
    for j, theta in enumerate(thetas.flatten()):
        x_lin_norm = (x_lin**j-means[j])/stds[j] 
        y_est = y_est + theta*(x_lin_norm)
    line.set_data(x_lin, y_est)
    poly_str = 'w_0'
    for u in range(i):
        poly_str = poly_str + '+w_{'+str(u+1)+'}x^{'+str(u+1)+'}'

    #ax[0].set_title(r'$%s$'%poly_str+'\n'+r'$J(\theta)=%0.0f$'%(J))
    ax[0].set_title(r'$%s$'%poly_str)
    ax[1].set_title(r'$J(\theta)=%0.0f$'%(J))
    ax[1].scatter(i,J, color='r')
    return line, ax


def get_polinomial_animation(X_LS, Y_LS, X_LS_test, Y_LS_test, model = get_parameters_polinomial_model, N = 40, lamb = 0, normalize = False, max_order = 9, fit_intercept=False, plot_test = True, interval = 200):
    x_lin = np.linspace(min([min(X_LS),min(X_LS_test)]),max(X_LS),1000)
    fig, ax = plt.subplots(2,1)

    plt.subplots_adjust(top=0.85)
    #fig.set_tight_layout(True)
    # Query the figure's on-screen size and DPI. Note that when saving the figure to
    # a file, we need to provide a DPI for that separately.
    print('fig size: {0} DPI, size in inches {1}'.format(fig.get_dpi(), fig.get_size_inches()))
    fig.set_size_inches(8, 6, True)
    # Plot a scatter that persists (isn't redrawn) and the initial line.
    ax[0].scatter(X_LS, Y_LS, color = 'b')
    #ax[0].set_xlim([-0.1-min(X_LS),max(X_LS)+0.1])

    losses_train= get_loss_polys(X_LS,Y_LS, model, max_order, lamb = lamb, normalize = normalize, fit_intercept=fit_intercept)
    #losses_test= get_loss_polys(X_LS_test,Y_LS_test, model, max_order, lamb = lamb, normalize = normalize, fit_intercept=fit_intercept)
    ax[1].plot(10*np.log(losses_train))
    #ax[1].plot(10*np.log(losses_test))

    #if plot_test:
    #    ax[0].scatter(X_LS_test, Y_LS_test, color = 'y')
    #    ax[0].set_ylim([-0.1-min(min(Y_LS), min(Y_LS_test)),max(max(Y_LS), max(Y_LS_test))+0.1])
    #else:
    #    ax[0].set_ylim([-0.1-min(min(Y_LS)),max(max(Y_LS))+0.1])

    line, = ax[0].plot([], [], 'k-')
    anim = FuncAnimation(fig, animate_poly, fargs=[ax, line, model, x_lin, X_LS,Y_LS, lamb , normalize, fit_intercept],frames=range(max_order+1), interval=interval, init_func=None, blit=False)
    return anim

def get_loss_function(X,y,w):
    error = y - X.dot(w)
    return np.log(error.T.dot(error))

def plot_loss_function(X, y, w0_est, w1_est, w0_range = [0,2], w1_range = [0, 4], N = 101, loss_fn = get_loss_function):
    y = y.reshape(len(y),1)
    w0s = np.linspace(*w0_range, N)
    w1s = np.linspace(*w1_range, N)
    loss_fuction = np.zeros((N,N))
    for iw0, w0 in enumerate(w0s):
        for iw1, w1 in enumerate(w1s):
            ws = np.array([[w0],[w1]])
            loss_fuction[iw0,iw1] = loss_fn(X,y, ws) 
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Xp, Yp = np.meshgrid(w0s, w1s)
    w_est = np.array([[w0_est],[w1_est]])
    loss_est = loss_fn(X,y, w_est)
    ax.scatter(w0_est,w1_est,loss_est , color='y', s = 100)
    ax.plot_surface(Xp, Yp, loss_fuction.T, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.set_xlabel('w0s')
    ax.set_ylabel('w1s')
    return w0s, w1s, loss_fuction

def plot_MSEs(MSEs, MSEs_test, labels = None, figsize=(12, 5), title = None):
    if MSEs_test is None:
        f, (ax_arr) = plt.subplots(1,1, sharex=False, sharey=False, figsize=figsize)
        ax_arr = [ax_arr]
    else: 
        f, (ax_arr) = plt.subplots(1,2, sharex=False, sharey=False, figsize=figsize)
    for i in range(len(MSEs)):
        ax_arr[0].plot(10*np.log(MSEs[i]), label = labels[i])
        if MSEs_test is not None:
            ax_arr[1].plot(10*np.log(MSEs_test[i]), label = labels[i])

    if title is not None:
    	ax_arr[0].set_title(title)
    else:
	    ax_arr[0].set_title('MSE de training set')
	    if MSEs_test is not None:
	        ax_arr[1].set_title('MSE de testing set')
    ax_arr[0].legend()
    ax_arr[1].legend()
    plt.show()