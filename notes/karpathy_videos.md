# Karpathy's videos

## Lecture 1:  The spelled-out intro to neural networks and backpropagation: building micrograd    

YouTube: https://www.youtube.com/watch?v=VMj-3S1tku0

* Learn to use Jupyter notebooks at last
* Autograd (Automatic gradient) engine implements backpropagation algorithm
* Every ANN training framework still uses backprop
* Backprop can be used for any kinds of directed graphs without loops, not only ANNs 
* Numpy - numerical library for Python (what is a good alternative for Scala?)
* Backprop uses Chain Rule (look up on Wikipedia) to calculate gradients for nodes distant more than 1 layer
* We calculate gradients for each node and then we know how we can increase or dcrease the total result by adding (a value * each local gradient) to the given input
* Karpathy uses tanh as ReLU and it seems he's ok with signals being <0.0 (so, the range is < -1.0, 1.0 > )
* He speaks briefly about building topological sorted graphs, because if we have a Directed Acyclic Graph, there is a simple algorithm that can do it and then visualisation is straightforward.
* Backprop is actually a bit more complicated than this simple version, because this simple version assumes the graph is really a tree, not a graph. When one node propagates the value to two subsequent nodes, then the backprop will ignore one of them.
  *  The solution is to replace the simple chain rule with multivariant chain rule version. The partial gradients need to be added to become a total gradient at a given node.
* PyTorch works on tensors
  * PyTorch is overcomplicated and in Python
  * It has backprop (.backward()) already implemented
* In Karpathy's model, Neuron has:
  * weights (ws: Vector[Double])
  * bias signal b (: Double) such that for the input vector ins the output is activation function of `ins*ws+b`
*   Layer is a vector of neurons
*  MLP (Multi-Layered Perceptron) is a vector of Layers
* The "loss" is a difference (a distance) of squares  between the ideal response for a give set of inputs and what we actually get. `loss = sum((predicted(i)-output(i))^2)`
  * A skoro tak to możemy użyć backprop do policzenia jak kolejne node'y wpływają na wartość loss. 
  * Ideally, the total loss would be a sum of losses for each pair of prepared inputs and outputs. But for really big networks it's more efficient to select a subset of those pairs that kinda sorta represents the whole set and calculate the total loss only on them. 
* Training an MLP means decreasing its loss.
  * For that we take each parameter (the weight of a connection between neurons and each bias signal) and change it by a small negative amount (e.g. -0.01) times its gradient.
  * If the loss is big, we can start with a bigger amount (-0.1 maybe) and then gradually lower it. This is called "learning rate decay".
  * Using too large amount risks that we will overshoot and make loss bigger because the parameters will be moved too much in the other direction
  * There are other ways to calculate loss
  * There's also something called "L2 regularization" which looks like a small adjustment of total loss depending on how active the whole network is, ie. if the parameters are overall closer to 1.0 or -1.0 than to 0.0. I helps to fight against overfitting.
    * alpha = 0.00001
    * reg_loss = alpha * sum(parameters(i) ^ 2)
    * total_loss = loss + reg_loss
  * 
* That one weird tweet. Most common neural net mistakes:
  * you didn't try to overfit a single batch first
  * you forgot to toggle train/eval mode for the net
  * you forgot to .zero_grad() (in pytorch) before .backward()
  * you passed softmaxed outputs to a loss that expected raw logits
* Here Karpathy made the error number 3. Since he used multi-value gradient - i.e. the one that adds partial gradients - he should have reset and recalculate gradients between each training iteration. He didn't do it, so gradients computed in the subsequent iteration were added to the already existing gradients from the previous iteration.

### Gobblydoktalk:

* Directed Acyclic Graph (DAG) - a directed graph without loops

* A ReLU, or rectified linear unit, is a type of nonlinear activation function commonly used in neural networks.
  * so, a sigmoid function or a max(0, x) function
  * https://builtin.com/machine-learning/relu-activation-function

* Scalar - just a number

* Forward pass - The action of giving inputs to a graph and running the calculations until we have the results.

* Gradient - a derivative of how a change to a given input affects the total result

* Local gradient - a derivative of how a change to a given input affects the result at the next node (that is an input to the yet next node, and so on, to the total result)

* Neuron bias - additional signal added to the sum of weights*signals computed by the neuron before giving it to ReLU.

* A tensor - An n-dimensional vector. Basically a generalized matrix. (afaik, tensors with n > 4 make no sense)

* Parameters - When you see a statement that an LLM has "x billion parameters", those parameters are:

  * For each neuron: all its weights + its bias signal
  * For each layer: all parameters of all its neurons
  * For an MLP: all parameters of all its layers
  * For the LLM: all parameters of all its MLPs + coefficients used to propagate outputs of one MLP to inputs of another (I suspect their number is negligible at this point)

  