# simple-levenberg-marquardt-torch
A simple implementation of the Levenberg-Marquardt algorithm in PyTorch. (Works for functions fine, but is still work in progress)

## Example

```
def fun1(x: torch.Tensor, p: torch.Tensor):
    return p[4] * x**5 + p[3] * x**4 + p[2] * x**3 + p[1] * x**2 + p[0] * x
    
# Generate some artificial data
x = torch.linspace(-3, 3, 100)
p = torch.Tensor([2, -1, 0.5, 3, 10])

artificial_data = fun1(x, p) + torch.rand(100) / 2

LM = LevenbergMarquardt(fun1, beta0=torch.ones(5))

print(LM)

beta, loss = LM.fit(x, artificial_data)
```

## Levenberg-Marquardt (doc)

The first optimizer is the Levenberg-Marquardt method, which is based on the Gauss-Newton method and thus finds the optimium in a local area.
The motivation of this algorithm is the least squares method for nonlinear functions f.
As the objective function, the problem is defined here as follows:

$$\\mathrm{argmin}_\\beta S(\\beta) = \\sum_{i=1}^m [y_i - f(x_i, \\beta)]^2$$

Therefore, equation (1) considers the squared error between an arbitrary function $f$ and the data points $y_i$ over the domain $x_i$.
The algorithm then iteratively calculates a new $\delta$ for each step, which determines the new parameter values by adding them to $\beta$
(Update Step: $\beta_{\tau + 1} = \beta_{\tau} + \delta$). Here, the update vector is calculated by the following equation:

$$\\left(\\mathbf{J}^{\\mathrm{T}}\\mathbf{J} + \\lambda \\cdot \\mathrm{diag}\\left(\\mathbf{J}^{\\mathrm{T}}\\mathbf{J}\\right)\\right) \\mathbf{\\delta} = \\mathbf{J}^{\\mathrm{T}} \\left[\\mathbf{y} - \\mathbf{f}(\\mathbf{x}, \\mathbf{\\beta})\\right]$$

Problematic about this step can be the Hessian matrix $\mathbf{J}^{\mathrm{T}}\mathbf{J}$. If you divide by too high / too low values, `NaN` values can occur within this matrix.
If this occurs, an error is raised. The vector $\delta$ is then added with the learning rate $\alpha$ to the $\beta$ of the previous iteration step.


### _class_ LevenbergMarquardt(func: callable, beta0: Tensor, alpha: float = 0.1, lmbda: float = 0.5, num_iter: int = 150)
Implements the Levenberg-Marquardt algorithm for fitting a costume function.
Problem:

$$\\mathrm{argmin}_\\beta S(\\beta) = \\sum_{i=1}^m [y_i - f(x_i, \\beta)]^2$$


* **Raises**

    
    * **TypeError** – if func is of wrong type


    * **TypeError** – if beta0 is of wrong type


    * **TypeError** – if alpha is of wrong type


    * **TypeError** – if lmbda is of wrong type


    * **TypeError** – if num_iter is of wrong type


    * **RuntimeError** – if func doesn’t have the right amount of arguments (=2)


    * **ValueError** – If Jacobian matrix contains NaN values


    * **ValueError** – If loss can’t be calculated



* **Returns**

    if printed returns nice string



* **Return type**

    str



#### fit(x: Tensor, y: Tensor, verbose: int = 1)
Tries to find a suitable set of parameters


* **Parameters**

    
    * **x** (*torch.Tensor*) – x axis points


    * **y** (*torch.Tensor*) – data


    * **verbose** (*int**, **optional*) – what will be printed out. Defaults to 1.



* **Returns**

    params, last loss



* **Return type**

    tuple
