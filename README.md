# AI_Projects
History of AI Projects in Python Programming Language


     The naïve part of a Naïve Bayes Classifier is the fact that it assumes all variables/features to be stochastically independent of each other, meaning one variable does not affect the probability of the other.

     A stochastic recurrent neural network.  Recurrent networks have connections between nodes(neurons) that form directed cycles.  This is a different type of network than feed forward networks.  Weights are usually stored in matrix form.  Recurrent networks can take arbitrary numbers of inputs.  Stochastic networks are made by introducing random variations into the network.  This helps them avoid the problem of getting stuck in local minimums.
     
     Boltzmann Machines consist of visible units, hidden units, weights, energy, temperature, and a learning rate.  Training occurs through a gradience descent approach and takes place in two phases.  In the phase+­, the visible vector is clamped according to a state vector from the training set.  In phase-, The network runs freely with no state clamping.
     There is a global E defined for the network that is the negative of the sum of weights times the state of unit i and j, plus the sum of the bias of i times the state of unit i.
     Temperature is a scalar that modifies the sigmoid function so that as it approaches zero, the sigmoid goes from a continuous function ranging from 0 to 1, into simulating a step-function, with an abrupt change from 0 to 1 when passing zero.
     Simulated annealing is a global optimization method where you set the Temperature parameter high and lower it as you the network runs until you reach a value called thermal equilibrium.  This is in hopes that you can find the global minimum.

     A Boltzmann Learning problem starts with a network of neurons/nodes that can be seen as having a state si = +1/-1.  There is a weight matrix connecting any two nodes i and j with individual weights being wij.  There is an energy function that is E = -1/2(sum[i,j =1 to N: wijsisj]
     A Kullback-Leibler divergence is used to obtain a value called G in which gradient descent is used on.  The gradient with respect to an individual weight is given by the equation, par(G)/par(wij) = 1/R[p­+ij – p-ij].  p+ij is the probability of i and j being on at thermal equilibrium in phase+, and p-ij is the same for phase-.
     A “learning/unlearning” method supervises the learning and makes sure the network doesn’t get completely off track.  The problem stems from not all local minima representing stored information.  Unlearning is used to remove these extra minima. 

     In a cost function for a neural network classifier the choice of error functions affects the gradient and the overall accuracy of the prediction.  The two are close but Cross Entropy Error is considered the better is considered better for classification and predictions problems during training, the problem being Mean Squared Error relying too much on the incorrect outputs.    Afterward classification is the error method chosen because that is the one sought after in the first place. 
     Mean Squared Error is the sum of your desired output minus your actual output squared, divided the the number of outputs.
Cross Entropy is the negative of the sum from t=0 to n of the natural log of the output vector times the desired or “target” output.

     Weights are more or less chosen randomly to get the best spread but there are ways to make sure that the linear region of the sigmoid is activated more often.  If the weights are too large or too small then the gradients will be small making learning slower.  The benefits of medium sized weight choices are that they are big enough to not grind learning to a halt, and the whole network will learn the linear part first.

	y' = y * (1 – y).
    • Y = 1 / (1 + e-x)
    • Y’ = (1 / (1 + e-x))2 d/dx(1+e-x)
    • Y’ = (1 / (1 + e-x)) (1 / (1 + e-x))(-e-x)
    • Y’ = (-e-x / (1 + e-x)) (1 / (1 + e-x))
        ◦ (-e-x / (1 + e-x)) * y
        ◦ ((1 + e-x) / (1 + e-x) - 1 / (1 + e-x))  * y
        ◦ (((1 + e-x) – 1) / (1 + e-x) * y
    • Y’ = y * (1 – y)

     If given a noisy input pattern a Hopfield net will converge to a trained pattern that matches the input pattern as well as possible.  Although a Hopfield net will always converge.  It may not converge to the correct thing.  Another problem could be the size of the data set itself.  If you have N for character recognition, you have to have N2 weights.  A final problem could be that there isn’t enough difference in the data to recognize any difference in the patterns.  Patterns that end up close together(characters that look more alike) may get confused for each other.  Maybe some type of bias or adaptive learning rate may help or some type of adaptation to the algorithm that attempts to check when similar patterns could be multiple things.

"1. Set the weights to small random values (e.g. between -.1 and .1)

2. Set the learning rate a << 1.

3. Repeat // until training error is low enoughSet total squared error = 0;For each training example e begin 
for j=0 to N do w[j] = w[j] + a*(d[e] - o[e])*x[j,e]; // x[0,e] is always 1
error = error + square(d[e] - o[e]);enduntil error < desired_value.

4. Store weight vector w in a file."


     Planned enhancements were an adaptive learning rate, confining the random weights so that they don’t end up too big or small, and refining the error function and finding which may return the best results(perhaps cross entropy?).

