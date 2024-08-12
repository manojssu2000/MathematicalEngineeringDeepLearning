# MathematicalEngineeringDeepLearning
Material for The Mathematical Engineering of Deep Learning. See the actual book content on [deeplearningmath.org](https://deeplearningmath.org) or (when it is out) purchase the book from CRC press.

This repository contains general supporting material for the book.

Below is a detailed list of the source code used for creating figures and tables in the book. We use [Julia](https://julialang.org/), [Python](https://www.python.org/), or [R](https://www.r-project.org/) and the code is sometimes in stand alone files, sometimes in [Jupyter](https://jupyter.org/) notebooks, sometimes as [R Markdown](https://rmarkdown.rstudio.com/), and sometimes in [Google Colab](https://research.google.com/colaboratory/). Many of our static illustrations were created using [TikZ](https://texample.net/tikz/examples/) by [Ajay Hemanth](https://www.linkedin.com/in/ajayhemanth/) and Vishnu Prasath with the [source of their illustrations](https://github.com/ajayhemanth/The-Mathematical-Engineering-of-Deep-Learning---TikZ) also available so you can adapt it for purposes. 

### Chapter 1
| Figure  | Topic       | Source Code  |
| ------- | ----------- | -----------  |
| 1.1     | Fast.ai example | [Python Google Colab](https://colab.research.google.com/drive/1YOjnlAqY71PspLn0QzoYl5SmcEmXr4GP?usp=sharing) |  
| 1.3     | Architectures   | [TikZ(a)](), [TikZ(b)](), [TikZ(c)](), [TikZ(d)](), [TikZ(e)](), [TikZ(f)](), [TikZ(g)](), [TikZ(h)]() |  
| 1.4     | Neurons         | [TikZ(b)](), [TikZ(d)]() |  
| 1.5     | Data on earth   | [Julia](Julia/data_world_in_zb.ipynb) |  

### Chapter 2
| Figure | Topic           | Source Code        |
| ------ | --------------- | -----------        |
| 2.1    | Supervised Learning | [TikZ]() |  
| 2.2    | Unsupervised Learning | [TikZ]() |  
| 2.3    | Simple regression | [R](R/Simple_Regression.R) |  
| 2.4    | Breast Cancer ROC curves | [R](https://github.com/yoninazarathy/MathematicalEngineeringDeepLearning/blob/master/R/Breast_Cancer_ROC_curves.R) |  
| 2.5    | Least Squares | [TikZ]() |  
| 2.6    | Loss functions | [Julia](Julia/LossFunctions.ipynb) |  
| Table 2.1 | Linear MNIST classification | [Julia](Julia/LinearMNIST_3_ways.ipynb) |
| 2.7    | Gradient Descent Learning Rate  | [Python](Python/Learning-Rate-Matters-GD-linear.ipynb) |  
| 2.8    | Loss Landscape  | [R](https://github.com/yoninazarathy/MathematicalEngineeringDeepLearning/blob/master/R/Loss_Landscape.R) |  
| 2.9    | Generalization and Training | [TikZ]() or [Julia](Julia/Expected_Performance_Curves.ipynb) |  
| 2.10   | Polynomial fit | [R](https://github.com/yoninazarathy/MathematicalEngineeringDeepLearning/blob/master/R/Polynomial_fit.R) |  
| 2.11   | K-fold cross validation | [TikZ]() |  
| 2.12   | K-means clustering | [R](https://github.com/yoninazarathy/MathematicalEngineeringDeepLearning/blob/master/R/kmeans-clustering.R) |  
| 2.13   | K-means image segmentation | [R](https://github.com/yoninazarathy/MathematicalEngineeringDeepLearning/blob/master/R/K-means-image-segmentation.R) |  
| 2.14   | Breast Cancer PCA | [R](https://github.com/yoninazarathy/MathematicalEngineeringDeepLearning/blob/master/R/Breast_Cancer_PCA.R) |  
| 2.15   | SVD Compression | [Julia](Julia/SVD_compression.ipynb) |

### Chapter 3
| Figure | Topic           | Source Code          |
| ------ | --------------- | -----------          |
| 3.1 and 3.2    | Logistic regression model curves and boundary | [R](R/R-code-logistic-smooth-link.R)  |        
| 3.3    | Components of an artificial neuron | [TikZ]() |
| 3.4    | Loss landscape of MSE vs. CE on logistic regression | [Python](Python/loss_landscapes_logistic.py) |
| 3.5    | Evolution of gradient descent learning in logistic regression | [R(a,b) First file](R/Gradient_Descent_logistic.R), [R(a,b) Second file](R/Function-for-Gradient-Descent-Logistic.R) |
| 3.6    | Shallow multi-output neural network with softmax | [TikZ]() |
| 3.7    | Multinomial regression for classification | [R](R/Figure-softmax-boundary-4-classes.R) |
| Table 3.1 | Different approaches for creating an MNIST digit classifier. | [Julia]() |
| 3.8    | Feature engineering in simple logistic regression | [R](R/R-code-logistic-beyond-linearity.R) |
| 3.9    | Non-linear classification decision boundaries with feature engineering in logistic regression | [R](R/R-code-Figure-versatile-Boundaries.R) |
| 3.10   | Non-linear classification decision boundaries with feature engineering in multinomial regression | [R](R/Figure-softmax-boundary-4-classes.R) same as 3.7 |
| 3.11 | Single hidden layer autoencoder | [TikZ]() |
| 3.12 | Autoencoder projections of MNIST including using PCA | [R](R/R-code-section-autoencoder.R) [TikZ]() |
| 3.13 | Manifolds and autoencoders | [R](R/simple-autoencoder-experiment.R) [TikZ]() |
| 3.14 | MNIST using autoencoders | [R](R/R-code-section-autoencoder.R) same as 3.12 |
| 3.15 | Denoising autoencoder | [TikZ]() |
| 3.16 | Interpolations with autoencoders | [R](R/R-code-section-autoencoder.R) same as 3.12, [Julia]() |


### Chapter 4
| Figure | Topic           | Source Code        |
| ------ | --------------- | -----------        |
| 4.1    | Convexity and local/global extrema   |    [Python](Python/Convexity_and_local_extrema.py)  |  
| 4.2    | Gradient descent with fixed or time dependent learning rate |  [Python](Python/GD-with-timde-dependent-alpha.py)  | 
| 4.3    | Stochastic gradient descent |  [Python]()  | 
| 4.4    | Early stopping in deep learning |  [Julia]()  | 
| 4.5    | Non-convex loss landscapes |  [Python]()  | 
| 4.6    | Momentum enhancing gradient descent |  [Python]()  | 
| 4.7    | The computational graph for automatic differentiation |  [TikZ]()  | 
| 4.8    | Line search concepts |  [Python]()  | 
| 4.9    | The zig-zagging property of line search | [Python]()  | 
| 4.10    | Newton's method in one dimension | [Python]()  | 


### Chapter 5
| Figure | Topic           | Source Code        |
| ------ | --------------- | -----------        |
| 5.1    | Fully Connected Feedforward Neural Networks | [TikZ(a)](), [TikZ(b)]()   |  
| 5.2    | Arbitrary function approximation with neural nets  | [TikZ(a)](), [Julia(b,c)]()   |  
| 5.3    | Binary classification with increasing depth |  [R](R/Figure_expressivity.R) |
| 5.4    | A continuous multiplication gate with 4 hidden units | [TikZ]()  | 
| 5.5    | A deep model with 10 layers | [TikZ]()  |  
| 5.6    | Several common scalar activation functions |  [Julia(a,b)]()  |  
| 5.7    | Flow of information in general back propagation | [TikZ]()  |  
| 5.8    | Simple neural network hypothetical example | [TikZ]() |  
| 5.9    | Flow of information in standard neural network back propagation | [TikZ]() |  
| 5.10   | Computational graph for batch normalization | [TikZ]() |  
| 5.11   | The effect of dropout  | [TikZ]()                    |  

### Chapter 6
| Figure | Topic           | Source Code          |
| ------ | --------------- | -----------          |
| 6.2    | VGG19 architecture | [TikZ]()        |  
| 6.3    | Convolutions | [TikZ(a)](), [TikZ(b)]() |  
| 6.6    | Convolution padding | [TikZ]() |  
| 6.7    | Convolution stride | [TikZ]() |  
| 6.8    | Convolution dilation | [TikZ]() |  
| 6.9    | Convolution input channels | [TikZ]() |  
| 6.10   | Convolution output channels | [TikZ]() |  
| 6.11   | Pooling | [TikZ(a)](), [TikZ(b)]() |  
| 6.13   | Inception module | [TikZ]() |  
| 6.14   | Resnets | [TikZ]() |  
| 6.17   | Siamese network | [TikZ]() |  


### Chapter 7 
| Figure | Topic           | Source Code        |
| ------ | --------------- | -----------        |
| 7.1    | Sequence RNN tasks | [TikZ(a)](), [TikZ(b)](), [TikZ(c)](), [TikZ(d)]() |
| 7.2    | Sequence RNN input output paradigms | [TikZ(a)](), [TikZ(b)](), [TikZ(c)](), [TikZ(d)]() |
| 7.3    | RNN recursive graph and unfolded graph | [TikZ]() |
| 7.4    | RNN unit | [TikZ]() |
| 7.5    | RNN language prediction training | [TikZ]() |
| 7.6    | Backpropagation  through time | [TikZ]() |
| 7.7    | Alternative RNN configurations | [TikZ(a)](), [TikZ(b)]() |
| 7.8    | LSTM and GRU | [TikZ(a)](), [TikZ(b)]() |
| 7.9    | Encoder decoder architectures | [TikZ(a)](), [TikZ(b)]() |
| 7.10   | Encoder decoder with attention | [TikZ]() |
| 7.11   | Attention weights | [TikZ]() |
| 7.12   | Flow of information with self attention | [TikZ]()  |
| 7.13   | Multi-head self attention | [TikZ]() |
| 7.14   | Positional embedding | [Julia(a,b)]() |
| 7.15   | Transformer blocks | [TikZ(a)](), [TikZ(b)]() |
| 7.16   | Transformer encoder decoder architecture | [TikZ]() |
| 7.17   | Transfomer auto-regressive application | [TikZ]() |


### Chapter 8
| Figure | Topic           | Source Code        |
| ------ | --------------- | -----------        |
| 8.1   | Generative modelling | [TikZ]()  |
| 8.2   | Variational autoencoder | [TikZ]() |
| 8.4   | Diffusion encoder and decoder | [TikZ]() |
| 8.6   | GAN architectures | [TikZ]() |
| 8.7   | Separation of GAN distributions | [TikZ]() |
| 8.8   | Wasserstein distance | [TikZ]() |
| 8.9   | Reinforcement learning | [TikZ]() |
| Equation (8.72) | An MDP optimal policy  | [Julia]() |
| 8.10   | Applications of GNN | [TikZ(a)](), [TikZ(b)](), [TikZ(c)]() |
| 8.11   | Directed and undirected graphs | [TikZ(a)](), [TikZ(b)]() |
| 8.12   | Transductive inductive learning | [TikZ(a)](), [TikZ(b)]() |
| 8.13   | Types of GNN tasks | [TikZ(a)](), [TikZ(b)](), [TikZ(c)]() |
| 8.14   | Aggregation in message passing  | [TikZ(a)](), [TikZ(b)]() |