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

* interactive jupyter
#+BEGIN_SRC ipython :results drawer raw
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
#+END_SRC

#+RESULTS:
# Out[12]:

* ob-ipython test
#+BEGIN_SRC ipython :results drawer raw
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

plt.hist(np.random.randn(20000), bins=200)
#+END_SRC

#+RESULTS:
# Out[8]:
#+BEGIN_EXAMPLE
  (array([  2.,   0.,   0.,   1.,   0.,   2.,   1.,   0.,   2.,   4.,   2.,
  3.,   2.,   3.,   1.,   3.,   4.,   3.,   3.,   6.,  13.,  10.,
  7.,   4.,  13.,   5.,  14.,  11.,  18.,  16.,  18.,  13.,  22.,
  20.,  27.,  26.,  32.,  38.,  31.,  37.,  44.,  46.,  42.,  33.,
  50.,  56.,  53.,  60.,  54.,  87.,  70.,  72.,  88., 111.,  91.,
  93., 103., 114., 114., 119., 134., 127., 145., 141., 175., 158.,
  188., 181., 191., 185., 222., 178., 237., 232., 221., 264., 244.,
  242., 219., 243., 233., 251., 269., 298., 250., 278., 297., 261.,
  281., 277., 270., 325., 289., 285., 282., 285., 269., 277., 289.,
  277., 283., 285., 280., 307., 253., 270., 280., 272., 260., 253.,
  256., 240., 275., 255., 227., 198., 227., 221., 194., 191., 189.,
  169., 183., 171., 147., 153., 148., 137., 139., 144., 133., 110.,
  108., 104., 101.,  83., 101.,  95.,  97.,  74.,  79.,  72.,  71.,
  64.,  50.,  54.,  51.,  40.,  49.,  39.,  33.,  36.,  23.,  31.,
  26.,  23.,  21.,  28.,  17.,  16.,  17.,  16.,  12.,  14.,   9.,
  13.,   6.,  13.,   7.,   8.,   5.,  10.,   6.,   1.,   6.,   2.,
  4.,   4.,   3.,   4.,   0.,   1.,   1.,   3.,   4.,   3.,   0.,
  0.,   2.,   0.,   0.,   0.,   2.,   0.,   0.,   0.,   0.,   0.,
  0.,   1.]),
  array([-3.48038147e+00, -3.44371782e+00, -3.40705417e+00, -3.37039052e+00,
  -3.33372687e+00, -3.29706322e+00, -3.26039957e+00, -3.22373592e+00,
  -3.18707227e+00, -3.15040862e+00, -3.11374498e+00, -3.07708133e+00,
  -3.04041768e+00, -3.00375403e+00, -2.96709038e+00, -2.93042673e+00,
  -2.89376308e+00, -2.85709943e+00, -2.82043578e+00, -2.78377214e+00,
  -2.74710849e+00, -2.71044484e+00, -2.67378119e+00, -2.63711754e+00,
  -2.60045389e+00, -2.56379024e+00, -2.52712659e+00, -2.49046294e+00,
  -2.45379929e+00, -2.41713565e+00, -2.38047200e+00, -2.34380835e+00,
  -2.30714470e+00, -2.27048105e+00, -2.23381740e+00, -2.19715375e+00,
  -2.16049010e+00, -2.12382645e+00, -2.08716281e+00, -2.05049916e+00,
  -2.01383551e+00, -1.97717186e+00, -1.94050821e+00, -1.90384456e+00,
  -1.86718091e+00, -1.83051726e+00, -1.79385361e+00, -1.75718997e+00,
  -1.72052632e+00, -1.68386267e+00, -1.64719902e+00, -1.61053537e+00,
  -1.57387172e+00, -1.53720807e+00, -1.50054442e+00, -1.46388077e+00,
  -1.42721712e+00, -1.39055348e+00, -1.35388983e+00, -1.31722618e+00,
  -1.28056253e+00, -1.24389888e+00, -1.20723523e+00, -1.17057158e+00,
  -1.13390793e+00, -1.09724428e+00, -1.06058064e+00, -1.02391699e+00,
  -9.87253338e-01, -9.50589689e-01, -9.13926040e-01, -8.77262391e-01,
  -8.40598742e-01, -8.03935093e-01, -7.67271444e-01, -7.30607795e-01,
  -6.93944146e-01, -6.57280497e-01, -6.20616848e-01, -5.83953199e-01,
  -5.47289550e-01, -5.10625901e-01, -4.73962252e-01, -4.37298604e-01,
  -4.00634955e-01, -3.63971306e-01, -3.27307657e-01, -2.90644008e-01,
  -2.53980359e-01, -2.17316710e-01, -1.80653061e-01, -1.43989412e-01,
  -1.07325763e-01, -7.06621142e-02, -3.39984653e-02,  2.66518364e-03,
  3.93288326e-02,  7.59924815e-02,  1.12656130e-01,  1.49319779e-01,
  1.85983428e-01,  2.22647077e-01,  2.59310726e-01,  2.95974375e-01,
  3.32638024e-01,  3.69301673e-01,  4.05965322e-01,  4.42628971e-01,
  4.79292620e-01,  5.15956269e-01,  5.52619918e-01,  5.89283567e-01,
  6.25947216e-01,  6.62610864e-01,  6.99274513e-01,  7.35938162e-01,
  7.72601811e-01,  8.09265460e-01,  8.45929109e-01,  8.82592758e-01,
  9.19256407e-01,  9.55920056e-01,  9.92583705e-01,  1.02924735e+00,
  1.06591100e+00,  1.10257465e+00,  1.13923830e+00,  1.17590195e+00,
  1.21256560e+00,  1.24922925e+00,  1.28589290e+00,  1.32255655e+00,
  1.35922019e+00,  1.39588384e+00,  1.43254749e+00,  1.46921114e+00,
  1.50587479e+00,  1.54253844e+00,  1.57920209e+00,  1.61586574e+00,
  1.65252939e+00,  1.68919303e+00,  1.72585668e+00,  1.76252033e+00,
  1.79918398e+00,  1.83584763e+00,  1.87251128e+00,  1.90917493e+00,
  1.94583858e+00,  1.98250223e+00,  2.01916587e+00,  2.05582952e+00,
  2.09249317e+00,  2.12915682e+00,  2.16582047e+00,  2.20248412e+00,
  2.23914777e+00,  2.27581142e+00,  2.31247507e+00,  2.34913872e+00,
  2.38580236e+00,  2.42246601e+00,  2.45912966e+00,  2.49579331e+00,
  2.53245696e+00,  2.56912061e+00,  2.60578426e+00,  2.64244791e+00,
  2.67911156e+00,  2.71577520e+00,  2.75243885e+00,  2.78910250e+00,
  2.82576615e+00,  2.86242980e+00,  2.89909345e+00,  2.93575710e+00,
  2.97242075e+00,  3.00908440e+00,  3.04574805e+00,  3.08241169e+00,
  3.11907534e+00,  3.15573899e+00,  3.19240264e+00,  3.22906629e+00,
  3.26572994e+00,  3.30239359e+00,  3.33905724e+00,  3.37572089e+00,
  3.41238453e+00,  3.44904818e+00,  3.48571183e+00,  3.52237548e+00,
  3.55903913e+00,  3.59570278e+00,  3.63236643e+00,  3.66903008e+00,
  3.70569373e+00,  3.74235737e+00,  3.77902102e+00,  3.81568467e+00,
  3.85234832e+00]),
  <a list of 200 Patch objects>)
#+END_EXAMPLE
[[file:./obipy-resources/F8N3Ou.png]]
* cost function plot
  #+BEGIN_SRC ipython :results drawer raw
    import numpy as np

    def h_theta_x(X, thetas):
	"""
	input = (m x (n+1)),  ((n+1) x 1)
	output = (m x 1)
	"""
	return X @ thetas

    def cost_func(thetas, X, y):
	"""
	input = ((n+1) x 1), ((m) x (n+1)), (m x 1)
	output = np.float64
	"""
	return np.sum(np.power(h_theta_x(X, thetas) - y, 2), axis = 0) / (2 * len(y))

    my_training_set = np.array([0, 1,
	    1, 0,
	    2, 1,
	    3, 0]).reshape(4, 2)

    m = len(my_training_set)
    X = np.hstack((np.ones((m, 1)), my_training_set[:, 0][:, np.newaxis]))
    X

    y = my_training_set[:, 1][:, np.newaxis]
    y

    start = -10
    stop = 10
    step = 1

    n = 1 # num of features

    my_thetas_lst = []
    for theta0 in np.arange(start, stop, step):
	    for theta1 in np.arange(start, stop, step):
		my_thetas = np.array([theta0, theta1]).reshape((n+1), 1)
		my_thetas_lst.append(my_thetas)

    thetas = my_thetas_lst[0]

  #+END_SRC

  #+RESULTS:
  # Out[26]:
  #+BEGIN_EXAMPLE
    array([[1., 0.],
    [1., 1.],
    [1., 2.],
    [1., 3.]])
  #+END_EXAMPLE
  # Out[25]:
  #+BEGIN_EXAMPLE
    array([[1., 0.],
    [1., 1.],
    [1., 2.],
    [1., 3.]])
  #+END_EXAMPLE
  # Out[24]:
  #+BEGIN_EXAMPLE
    array([[1., 0.],
    [1., 1.],
    [1., 2.],
    [1., 3.]])
  #+END_EXAMPLE
  # Out[23]:
  #+BEGIN_EXAMPLE
    array([[1., 0.],
    [1., 1.],
    [1., 2.],
    [1., 3.]])
  #+END_EXAMPLE
  # Out[22]:
  #+BEGIN_EXAMPLE
    array([[1., 0.],
    [1., 1.],
    [1., 2.],
    [1., 3.]])
  #+END_EXAMPLE
  # Out[21]:
  #+BEGIN_EXAMPLE
    array([[1., 0.],
    [1., 1.],
    [1., 2.],
    [1., 3.]])
  #+END_EXAMPLE
  # Out[13]:
  #+BEGIN_EXAMPLE
    array([[1., 0.],
    [1., 1.],
    [1., 2.],
    [1., 3.]])
  #+END_EXAMPLE
  # Out[11]:
  #+BEGIN_EXAMPLE
    array([[1],
    [0],
    [1],
    [0]])
  #+END_EXAMPLE
  # Out[9]:
  : <module 'numpy' from '/Users/sanjay/.virtualenvs/mlm/lib/python3.7/site-packages/numpy/__init__.py'>
