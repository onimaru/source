(algebra-einstein-convention)=
# Algebra and the Einstein Summation Convention

Here we will define important concepts and quantities with an elegant notation. This notation can be used in conventional linear algebra, tensor algebra and so on. It is a very useful tool to understand better a lot of quantities and operations between vectors, matrices and tensors.

## Einstein Summation Convention

The convention is very simple, it summarizes equations in which indexes are summed. Notice we may have indexes bellow or above a letter. For example:

$$\sum_{i=0}^{2} x_{i}y^{i} = x_{0}y^{0} + x_{1}y^{1} + x_{2}y^{2}$$

That kind of equation can be very long to write when operations are made. The Einstein summation convention hides the summation symbol and we agree that repeated symbols are being summed. The above equation can be written as:

$$\sum_{i=0}^{2} x_{i}y^{i} = x_{i}y^{i}$$

Let us see other examples:

$$\sum_{i} A^{li}x_{i}y^{j}z_{k} = A^{li}x_{i}y^{j}z_{k}$$

$$\sum_{i}\sum_{j} A^{li}x_{i}y^{j}z_{k} = A^{li}x_{i}y^{j}z_{k}$$

Let us define a conventional column vector as $\boldtext{v} = v^{i}e_{i}$, where $e_{i}$ are the base vectors defined {ref}`here <definitions>`. For simplicity we hide the base representation and write the vector simply as $v^{i}$. Since there is no repeated indexes this expression mean we are refering to all components of the vector. For example: if $\boldtext{v} \in \mathbb{R}^{3}$ we have

$$v^{i} = \begin{pmatrix} v^{0} \\ v^{1} \\ v^{2} \end{pmatrix}$$

