from inspect import getfullargspec
import torch

# Levenberg Marquardt algorithm for function fitting
class LevenbergMarquardt:
    """Implements the Levenberg-Marquardt algorithm for fitting a costume function.
    Problem:
    
    .. math:: \mathrm{argmin}_\\beta S(\\beta) = \sum_{i=1}^m [y_i - f(x_i, \\beta)]^2
    
    Raises:
        TypeError: if func is of wrong type
        TypeError: if beta0 is of wrong type
        TypeError: if alpha is of wrong type
        TypeError: if lmbda is of wrong type
        TypeError: if num_iter is of wrong type
        RuntimeError: if func doesn't have the right amount of arguments (=2)
        ValueError: If Jacobian matrix contains NaN values
        ValueError: If loss can't be calculated

    Returns:
        str: if printed returns nice string
    """
    # Learning Rate
    alpha: float = 0.1
    
    # Parameters
    beta: torch.Tensor = None
    beta0: torch.Tensor = None
    
    # damping parameter
    lmbda: float = 0.5
    
    def __init__(self, func: callable, beta0: torch.Tensor, alpha:float = 0.1, 
                 lmbda: float = 0.5, num_iter: int=150) -> None:
        """Applies the Levenberg Marquardt algorithm to a given data row.

        Args:
            func (Callable): (Non-) linear function
            beta0 (torch.Tensor): starting vector
            alpha (float, optional): learning rate. Defaults to 0.1.
            lmbda (float, optional): damping parameter. Defaults to 0.5.
            num_iter (int, optional): Number of iterations, that will be done. Defaults to 150

        Raises:
            TypeError: if func is of wrong type
            TypeError: if beta0 is of wrong type
            TypeError: if alpha is of wrong type
            TypeError: if lmbda is of wrong type
            TypeError: if num_iter is of wrong type
            RuntimeError: if func doesn't have the right amount of arguments (=2)
        """
        # Copy function to var
        self.fun = func
        
        #####################################################
        #   Checks, if Parameters are of the right type     #
        #####################################################
        # If function has not the correct amount of parameters
        if not callable(self.fun):
            raise TypeError('\033[91m' + "[ERROR] " + '\033[0m' + f"Parameter 'func' expects to get a function, but got: {type(self.fun)}")
        
        if not isinstance(beta0, torch.Tensor):
            raise TypeError('\033[91m' + "[ERROR] " + '\033[0m' + "beta0 has to be of type 'torch.Tensor'!")
        
        if not isinstance(alpha, float):
            raise TypeError('\033[91m' + "[ERROR] " + '\033[0m' + "alpha has to be of type 'float'!")
        
        if not isinstance(lmbda, float):
            raise TypeError('\033[91m' + "[ERROR] " + '\033[0m' + "lmbda has to be of type 'float'!")
        
        if not isinstance(num_iter, int):
            raise TypeError('\033[91m' + "[ERROR] " + '\033[0m' + "num_iter has to be of type 'int'!")
        
        fun_args = getfullargspec(self.fun)
        if len(fun_args.args) != 2:
            string_args = ""
            for arg_name in fun_args.args:
                string_args += '"' + arg_name + '", '
            raise RuntimeError('\033[91m' + "[ERROR] " + '\033[0m' + "Passed function must have only 2 arguments: x-axis and y-function values. But got: " + string_args[:-2])
        
        
        # Copy Starting point and general parameters to class
        self.beta0 = beta0.float()
        self.alpha = alpha
        self.lmbda=lmbda
        
        # Get the amount of parameters we need to find
        self.num_params = self.beta0.shape[0]
        
        # Set iteration limit
        self.num_iter = num_iter
         
    def __repr__(self) -> str:
        return '\033[93m' + "[INFO] " + '\033[0m' + f"Levenberg-Marquardt-Algorithm(Params: {self.num_params}, #iteration: {self.num_iter})"
        
    def fit(self, x: torch.Tensor, y: torch.Tensor, verbose: int=1):
        """Tries to find a suitable set of parameters

        Args:
            x (torch.Tensor): x axis points
            y (torch.Tensor): data
            verbose (int, optional): what will be printed out. Defaults to 1.

        Returns:
            tuple: params, last loss
        """
        loss = 0
        self.beta = self.beta0

        for cnt in range(0, self.num_iter):  
            J = torch.autograd.functional.jacobian(self.fun, inputs=(x, self.beta))[-1]
            
            # Create diagonal matrix
            diagonal_matrix = torch.zeros((self.num_params, self.num_params)) * torch.max(torch.diag(J.T @ J))

            # Create 'damped version' Tensor                
            damped_version = J.T @ J + self.lmbda * diagonal_matrix
            if bool(torch.any(torch.isnan(damped_version))):
                raise ValueError('\033[91m' + "[ERROR] " + '\033[0m' + "Some values in the Jacobian matrix are NaN! Fitting failed!")
            
            delta = self.alpha * torch.inverse(damped_version) @ J.T @ (y - self.fun(x, self.beta))
            
            # Update (parameters) beta
            self.beta += delta
            
            # Calculate the new loss
            loss = torch.sum((y - self.fun(x, self.beta))**2)
            if bool(torch.isnan(loss)):
               raise ValueError('\033[91m' + "[ERROR] " + '\033[0m' + "Loss value couldn't be calculated (is NaN)!")
           
            if verbose > 0 and cnt % 25 == 0:
                print('\033[92m' + "[OK] " + '\033[0m' + f"Iteration {cnt}: loss = {float(loss)}")
                
        
        if verbose > 0:
            print('\033[92m' + "[OK] " + '\033[0m' + f"Finished! Final loss: {loss:.2f}")
                
        return self.beta, float(loss)


if __name__ == '__main__':  
    print('\033[92m' + "[OK] " + '\033[0m' + "finished!")
