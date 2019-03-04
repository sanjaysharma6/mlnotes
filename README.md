* bloomberg ml course
** david rosenberg the tutor himself says:
   I’m sure every topic in Foundations is taught in some other class
   somewhere. But here some highlights that might be of interest:
   discussion of approximation error, estimation error, and optimization
   error, rather than the more vague “bias / variance” trade off; full
   treatment of gradient boosting, one of the most successful ML
   algorithms in use today (along with neural network models); more
   emphasis on conditional probability modeling than is typical (you give
   me an input, I give you a probability distribution over outcomes —
   useful for anomaly detection and prediction intervals, among other
   things), geometric explanation for what happens with ridge, lasso, and
   elastic net in the [very common in practice] case of correlated
   features; guided derivation of when the penalty forms and constraint
   forms of regularization are equivalent, using Lagrangian duality (in
   homework), proof of the representer theorem with simple linear
   algebra, independent of kernels, but then applied to kernelize linear
   methods; a general treatment of backpropagation (you’ll find a lot of
   courses present backprop in a way that works for standard multilayer
   perceptrons, but don’t tell you how to handle parameter tying, which
   is what you have in CNNs and all sequential models (RNNs, LSTMs,
   etc.); in the homework you’d code neural networks in a computation
   graph framework written from scratch in numpy; well, basically every
   major ML method we discuss is implemented from scratch in the
   homework.
