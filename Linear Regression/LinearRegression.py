import numpy as np

def assert_size(X,y):
    """Verify if target and inputs have the same number of enteries"""
    #if condition returns False, AssertionError is raised:
    assert y.shape[0] == X.shape[0], "X and y not of same shape: They do not contain the same number of observations."
    
class Linear_Regression():
    """Linear Regression algorithm implemented with Batch Gradient Descent and Ordinary Least Squares cost optimization algorithms"""
    def __init__(self, optimizer):
        self.optimal_weights = None
        self.optimizer = optimizer
        
    def fit(self,X,y,epoch=10000,tolerance=0.0001,learning_rate=0.1):
        """Fit model to training data
        X: set of training attributes.
        y: Training Targets.
        optimizer: Cost Optimization Method for Linear Regression.
        epoch: No of iterations for weight optimization.
        tolerance: threshold for stopping gradient descent.
        learning_rate: tuning parameter that determines the step size at each iteration.
        """
        assert_size(X,y)
        if self.optimizer == "bgd":
            #Pad features with bias of 1
            X_new = np.c_[np.ones((X.shape[0],1)),X]
            y = y.reshape(-1,1)    #suitable shape of target for loss computation
            #Initialize random weights and intercept
            self.optimal_weights  = np.random.rand(X_new.shape[1],1)

            for iteration in range(epoch):
                #compute prediction 
                y_pred = X_new.dot(self.optimal_weights )

                #compute gradient of loss function with respect to weight
                gradient = (2/X_new.shape[0]) * X_new.T.dot(y_pred - y)

                #Update weight paramters in direction of decreasing gradients
                self.optimal_weights  = self.optimal_weights  - (learning_rate * gradient)
                #stop descent once in the neighborhood of global minima
                if np.linalg.norm(gradient) < tolerance:
                    break #Quit descent once the norm of cost gradient falls below threshold.
                    
            #Get intercept and coefficients from learned weights
            self.intercept_, self.coef_ = self.optimal_weights[-1],self.optimal_weights[:-1]

        elif self.optimizer == "ols":
            #Pad features with bias of 1
            X_new = np.c_[np.ones((X.shape[0],1)),X]
            y = y.reshape(-1,1)  #suitable shape of target for loss computation
            #Obtain optimal weights using the ols normal equation
            self.optimal_weights = np.linalg.inv(X_new.T.dot(X_new)).dot(X_new.T).dot(y)
            y_pred = X_new.dot(self.optimal_weights) 
            
            #Get intercept and coefficients from learned weights
            self.intercept_, self.coef_ = self.optimal_weights[-1],self.optimal_weights[:-1]
        
        else:
            raise ValueError("Invalid optimizer %s: expected any of `bgd` or `ols`." % repr(self.optimizer))
            
    #predict method
    def predict(self,X_test):
        """Predict on test data"""
        X_test_new = np.c_[np.ones((X_test.shape[0],1)),X_test]
        predictions = X_test_new.dot(self.optimal_weights)
        return predictions
    
    #evaluation (score) method
    def score(self,y_true,y_pred):
        """Generate loss (Mean Squared Error)"""
        assert_size(y_true,y_pred)
        #default is MSE error
        error = np.mean(((y_true - y_pred)**2))
        return error