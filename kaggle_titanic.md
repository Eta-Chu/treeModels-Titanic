# Tree Model

### 1. CART (Classification and Regression Tree)

​	In classification tree algorithm, we need a standard criteria to judge that the perfemence for a specific decision i.e. the "split". First of all, we introduce a value called "Gini Impurity" to discribe the impurity of a group of samples.

+ Gini Impurity

$$
G = 1 - \sum_{k} p_{k}^2
$$

where the $p_k$ is the "sample probability", in algorithm it's calculated by the frequency of label. For binary classification problems:
$$
\begin{aligned}
G &= 1 - p_0^2 - p_1^2 \\
&= 1 - p^2 - (1-p)^2 \\
&= 2p(1 - p)
\end{aligned}
$$
It's easy to seen that the maxmum value is getting in $p=1/2$, witch means that the group is most chaotic. Then we can using the "impurity difference" to judge whether the "splitting" is good or not. It's presented by:
$$
\Delta G = G_{parent} - G_{children} \\
G_{children} = \frac{N_L}{N} G_L + \frac{N_R}{N} G_R
$$
where the $G_{parent}$ and $G_{children}$ is the impurity before and after spilitting. $G_L$ and $G_R$ is the impurity of left and right branch respectively after splitting. For a good enough decision, the impurity should decrease which means that the bigger $\Delta G$ is, the better. So every decision in Decision tree algorithm is to find out the splitting which make 
$$
(f^*, t^*) = \text{argmax}_{f, t} \Delta G
$$
where the $f, t$ represent feature and threshold respectively.

> ##### Regression Tree
>
> As for Regression Tree algorithms, the spliting is very different compared with classification. First, we need to figure out what's the target for CART in regression task.
>
> For a regression problem, the core task is to find out a mapping $T(\{x \}) \rightarrow y$ according to all samples $\{ x\}$. For the tree model, it will spilt samples to different groups i.e different leafs $R_j$, and using the average value $\bar{y}_j$ to represent each leaf, like a peicewise function:
> $$
> T(x) = 
> \begin{cases}
> \frac{1}{N_1}\sum_{x_i \in R_1} y_i ,\  x\in R_1 \\
> \frac{1}{N_2}\sum_{x_i \in R_2} y_i ,\  x\in R_2 \\
> \cdots
> 
> \end{cases}
> $$
> where each group $R_j$ is so-called leaf node in CART. Notice that the leaf node is splited by feature $x$. Now the algorithm need a criteria to evaluate the quality of spliting, like "Gini Impurity deference". In regression tree, it's vary nature to using the sample-size-weighted mean squared error of the child nodes:
> $$
> \text{MSE}_{split} = \frac{N_L}{N_L + N_R}\text{MSE}_{L} + \frac{N_R}{N_L + N_R}\text{MSE}_R \\
> \text{MSE} = \frac{1}{N_j} \sum_{x_i \in R_j} (y_i - \bar{y}_j)^2
> $$
> And the way to find the best spilt in every level is same as classification tree:
> $$
> (f^*, t^*) = \text{argmin}_{f, t} \text{MSE}_{split}
> $$

### 2. Bagging(bootstrap aggregating)

+ bootstrap sampling:

$$
P = (1 - \frac{1}{N})^N,  \\
\text{lim}_{N \rightarrow \infty} P = \frac{1}{e} \sim 0.368
$$

+ variance:

  We can define the variance of the Forest $\text{Var}_{D}[T_D(x)]$
  $$
  \begin{aligned}
  \text{Var}[\frac{1}{M}\sum_{m=1}^M T_m(x)] &= \frac{1}{M^2} \Big(
  \sum_{m} \text{Var}[T_m(x)] + 2 \sum_{i<j}\text{Cov}[T_i(x), T_j(x)]
  \Big) \\
  &=\frac{1}{M^2} \Big(
  \sum_{m} \text{Var}[T_m(x)] + 2 \sum_{i<j}\sigma_{i} \sigma_{j}\text{Corr}[T_i(x), T_j(x)] 
  \Big) \\
  \end{aligned}
  $$
  if approximately set $\text{Var}(T_m(x)) \equiv \sigma^2$, then we have
  $$
  \begin{aligned}
  RHS &\sim \frac{1}{M^2}\Big(
  M \sigma^2 + 2 \frac{M(M-1)}{2} \sigma^2 \rho
  \Big) \\
  &= \frac{\sigma^2}{M}(1+\rho (M-1)) \\
  &= \rho \sigma^2 + \frac{1-\rho}{M} \sigma^2
  
  \end{aligned}
  $$
  when we increase the number of CART which means $M\rightarrow \infty$, the second term in variance will disappear, lefting the first term which is determined by correlation of different CART $\rho$. So in RF algorithm, the key point is to decrease the correlation between every tree by introducing *randomness*.

  

> $$
> \begin{aligned}
> \text{Var}(X) &= E[(x - E[x])^2] \\
> \text{Var}(aX) &= a^2 \text{Var}(X) \\
> \\
> \text{Var}(X_1 + X_2) &= E\Big[\Big((X_1 + X_2) - E[X_1+X_2]\Big)^2\Big] \\
> &= E\Big[\Big((X_1 - E[X_1]) + (X_2 - E[X_2])\Big)^2\Big] \\
> &= \text{Var}(X_1) + \text{Var}(X_2) + 2 \text{Cov}(X_1, X_2)\\
> \\
> \text{Var}(\sum_{i}X_i) &= \sum_{i}\text{Var}(X_i) + 2\sum_{i<j}\text{Cov}(X_i, X_j)\\
> 
> \text{Corr}(X_1, X_2) &= \frac{\text{Var}(X_1, X_2)}{\sigma_{X_1} \sigma_{X_2}}
> \end{aligned}
> $$

### 3. Boosting

Boosting is an ensemble method that builds a strong predictor by adding weak learners sequentially. Each new learner is trained using information from the current ensemble, so that it can improve the predictions made by the previous learners. For tree-based boosting, shallow decision trees are commonly used as the base learners, and their outputs are combined into an additive model:

$$
F_M(x) = F_0(x) + \sum_{m=1}^{M} \alpha_m h_m(x)
$$

Here $F_0$ is the initial prediction and $\alpha_m$ controls the contribution of the $m$th learner. Bagging trains learners on bootstrap samples and aggregates their predictions, primarily to reduce variance; boosting instead builds learners sequentially to reduce the ensemble's loss. AdaBoost emphasizes samples that the current ensemble classifies poorly by updating their weights. Gradient Boosting fits new learners to the negative gradient of a chosen loss function, while XGBoost uses a regularized second-order approximation to evaluate tree splits and leaf values. These are different ways to decide what the next learner should contribute. [3–5]

##### a. AdaBoost(Adaptive Boosting)

​	Let's briefly introduce the procedure of AdaBoost algorithm. The core idea of AdaBoost is adaptively adjusting the weight of each tree and training every tree according to the perfermance of former trees. It's realized by two kinds of 'weight'. For the first, each sample is not considered to be equally weighted in AdaBoost. In AdaBoost algorithm, it trains $M$ trees sequentially. In each training process, the algorithm changes the sample weight adptively. For example, when training CART $h_m(x)$, there is a trainingset weight set $\{w_i^m\}$. Then we have
$$
\begin{aligned}
G_{children} &= \frac{W_R}{W} G_R + \frac{W_L}{W} G_L \\
W_R &= \sum_{i\in R} w_i \\
W_L &= \sum_{i\in L} w_i \\
\end{aligned}
$$
And the error rate for $h_m(x)$ is 
$$
\epsilon_m = \frac{\sum_{i \in error\ set} w_i^m}{W}
$$
Latter we will introduce how to confirm the value of $\{ w_i^m \}$. For the second 'weight', the final decision is weighted conbination of the whole forest
$$
F_M(x) = \sum_{m=1}^M \alpha_m h_m(x)
$$
where the $\alpha_m$ is the weight of $m$th tree $h_m(x)$. In random forest, the $\alpha$ is seen as the same value. The weight of $h_m(x)$ can be decided by the error rate $\epsilon_m$ for whole training set, which means if a tree has a bad perfermance for training set, it's wight will be set to be small. In summary, the procedure of AdaBoost can be summarized by 
$$
\{w_i^m\} \rightarrow h_m \rightarrow \epsilon_m \rightarrow \alpha_m \rightarrow \{ w_i^{m+1}\}
$$
For the first tree, the $\{w_i^m \}$ can be set with same value. The update equations is 
$$
\alpha_m = \frac{1}{2} \ln \frac{1-\epsilon_m}{\epsilon_m}
$$

$$
w_{i}^{m+1} \leftarrow \frac{w_i^m \exp[-\alpha_m y_i h_m(x_i)]}{Z_m}
$$

​	In the above, we didn't explain why the update equations take this form. Now let's start this with the "exponential loss" in $k$th step:
$$
L(F_k) = \sum_{i=1}^N e^{-y_i F_k(x_i)}
$$
In binary classification problems, the $y_i = -1\ or\ +1$ is the label of sample and the class decided by $F_k$ is $\hat{y}_i = \text{sign}(F_k(x_i))$. So if $F_k$ has a good classification for sample $x_i$, the $\hat{y}_i$ will have the same sign as $y_i$, then the contribution to loss for $x_i$ is small. This is why we use exponential loss here. Notice that
$$
F_k(x_i) = F_{k-1}(x_i) + \alpha_k h_k(x_i)
$$
So we have
$$
L(F_k) = \sum_i e^{-y_iF_{k-1}(x_i)} e^{-\alpha_k y_i h_k(x_i)}
$$
**we can define the $w_i^{k} \equiv e^{-y_i F_{k-1}(x_i)} / Z_k$** (here the $Z_k$ is the normalization factor), which means the wight for sample $x_i$ in trainning $k$th CART $h_k$ is the exponent loss of $k-1$th Forest. As we can see from the definition if apprent "Forest" $F_{k-1}$ has poor performance for sample $x_i$, AdaBoost algorithm will increase the weight of sample $x_i$ in training procedure of $h_k$.

Let's back to the equation $L(F_k) = \sum_i w_i^k e^{-y_i \alpha_k h_k(x_i)}$, it's not hard to notice that only $\alpha_k$ is the undetermined parameter. And the direct way to confirm $\alpha_k$ is to minimize $L(F_k)$, i.e. the solution of
$$
\frac{d L}{d \alpha_k} = \sum_i -y_i h_k(x_i) w_i^k e^{-y_i \alpha_k h_k(x_i)} = 0
$$
This equation can be simplified further. Notice that
$$
y_i h_k(x_i) = 
\begin{cases}
+1, & correct \\
-1, & wrong
\end{cases}
$$
according this we can divide all samples to $c$ set (correct samples) and $\varepsilon$ set (wrong samples), and now the error $\epsilon$ of $h_k$ is 
$$
\epsilon_k = \sum_{i \in \varepsilon} w_i^k \\
\sum_{i \in c} w_i^k = 1 - \epsilon_k
$$
then we have
$$
\sum_{i \in c} - w_i^k  e^{-\alpha_k} + \sum_{i \in \varepsilon} w_i^k e^{\alpha_k} = 0\\
-(1-\epsilon_k) e^{-\alpha_k} + \epsilon_k e^{\alpha_k} = 0
$$
it's easy to get that the solution is 
$$
\alpha_k = \frac{1}{2} \ln \frac{1-\epsilon_k}{\epsilon_k}
$$
This is the update equation for tree weight $\alpha_k$. Last, let's simplify the definition of sample weight
$$
w_i^{k} \equiv e^{-y_i F_{k-1}(x_i)} / Z_k = \frac{e^{-y_i F_{k-2}(x_i)} e^{-y_i \alpha_{k-1} h_{k-1}(x_i)}}{Z_k} = \frac{w_i^{k-1} e^{-y_i \alpha_{k-1} h_{k-1}(x_i)}}{Z_k}
$$
This is the update equation for sample weight $w_i^k$. 

*results:*

<img src="/Users/eta/Library/Application Support/typora-user-images/image-20261113152122724.png" alt="image-20261113152122724" style="zoom:50%;" />

##### b. Gradient Boosting

​	The general process of GradientBoosting algorithm is as follows:

$$
F_{m-1}(x_i) \rightarrow r_{mi} = - \frac{\partial L}{\partial F_i}  \rightarrow T_m \rightarrow F_m(x_i)
$$
where the key updating is 

$$
F_m(x_i) = F_{m-1}(x_i) + \eta T_m(x_i)
$$


Let's start with a loss function $L(y_i, p(x_i))$, for binary classification problems, the loss function is cross entropy:
$$
L(y_i, p(x_i)) = -\Big[ y_i \log p(x_i) + (1-y_i)\log \big(1-p(x_i)\big)
\Big]
$$
in Gradient Boosting algorithms, generally $F(x_i)$ represents a logit rather than a probability, which means $p(x_i) = \sigma(F(x_i))$, $\sigma$ is the sigmoid function. Then we can rewrite the Loss as the function of $F$
$$
\begin{aligned}
L(y_i, F_i) &= -\Big[
y_i \log(\frac{e^{F_i}}{1+e^{F_i}})
+ (1 - y_i)\log(1 - \frac{e^{F_i}}{1 + e^{F_i}})
\Big] \\
&= -y_i F_i + \log(1 + e^{F_i})
\end{aligned}
$$
 Apprently, it's possible to calculat the gradient $r_i$ explicitly. It's 
$$
\begin{aligned}
r_i
&=-\frac{\partial L}{\partial F_i} \\
&= y_i - \sigma(F_i) \\
&= y_i - p(x_i)
\end{aligned}
$$
Next training a regression tree $T_m$ according to the map $\{x_i\} \rightarrow \{r_i\}$. Then update the final prediction $F_m(x_i)$ using $T_m$ and learning rate. **But actually in Gradient Boosting, the real situation is more complicated**. After getting the gradient regression tree $T_m$, it's possible to further get the optimal update for loss function. In orther words, the final logit can be rewritten as
$$
F_{m}(x_i) = F_{m-1}(x_i) + \eta \gamma_{mi}
$$
where the $\gamma_{mi}$ is the optimal update, it can be interpreted as $\gamma_{mi} = \Delta F \cdot r_{mi}$, the $\Delta F$ is the best step size. There is no need to calculate best step size. Instead of that, we can get the $\gamma_{mi}$ throught
$$
\gamma_{mj} = \text{argmin}_{\gamma} \sum_{x_i \in R_{mj}} L(y_i, \sigma(F_{m-1}(x_i) + \gamma))
$$
where the $R_{mj}$ is the leaf notes of $T_m$. Notice that in above equation, the subscript of $\gamma$ is $j$ not $i$, which means each leaf node have a same update value instead of each sample $x_i$. Before going into the details of how to find the value of $\gamma_{mj}$ for $R_{mj}$, let's briefly review the whole procedure of Gradient Boosting now. Once getting the $T_m$, we already have the fitted gradient. In the orther hand, the tree $T_m$ also divides samples  to different leafs when fitting the gradient. So instead of using gradient update directly, the algorithm choses to using the steepest descent. Actually, it can be understood that the final tree is replaced by a new tree:

$$
T_{m}^{\text{final}}(x_i) = 
\begin{cases}
\gamma_{m1}, \ x_i \in R_{m1}\\
\gamma_{m2}, \ x_i \in R_{m2}\\
\vdots \\
\gamma_{mJ}, \ x_i \in R_{mJ}
\end{cases}
$$

In summary, the whole procedure can be expressed as:
$$
F_{m-1}(x_i) \rightarrow r_{mi} = y_i - \sigma(F_{m-1}(x_i)) \rightarrow T_m \rightarrow R_{mj} \rightarrow \gamma_{mj}\rightarrow T_{m}^\text{final}(x_i) \rightarrow F_{m}(x_i)
$$
​	In the last, let's discuss how to determine the value of $\gamma_{mj}$ for a specific leaf $R_{mj}$ in tree $T_m$. First set the function needed to be optimized $f = \sum_{x_i \in R_{mj}} L(y_i, \sigma(F_{m-1}(x_i) + \gamma))$ and substitute the $L$ to cross entropy, 
$$
f = \sum_{x_i \in R_{mj}} \Big[
-y_i (F_i + \gamma) + \log(1 + e^{F_i + \gamma})
\Big]
$$
Then $\gamma_{mj}$ is the root of 
$$
\frac{d f}{d \gamma} = \sum_{x_i \in R_{mj}} \big(\sigma(F_{m-1}(x_i) + \gamma) - y_i\big) = 0
$$
Set $p(x_i, \gamma) = \sigma(F_{m-1}(x_i) + \gamma)$ and denote the above equaiton as $G(\gamma)$, i.e. the gradient function is
$$
G(\gamma) = \sum_{x_i \in R_{mj}} (p(x_i, \gamma) - y_i)
$$
If the root of $G(\gamma) = 0$ is $\gamma^\prime$ then it should satisfy $\sum p(x_i, \gamma^\prime) = \sum y_i$, which means the optimal correction $\gamma^\prime$ makes summation of predictional probability should equal to the number of samples belong to class "True" in leaf $R_{mj}$. It's very easy to understand. But calculating the solution of this equation is not easy in genetal. So in practice ones usually using numerical methods to find the solution approximately, such as **Newton-Raphson** method. In Newton-Raphson method, it gradually approaches the solution of the function $G(\gamma)$ using Taylor expansion to the first order
$$
G(\gamma) \simeq G(\gamma_0) + G^\prime(\gamma_0) (\gamma - \gamma_0)
$$
The $\gamma_0$ is the initial trial solution. In the end, it will get a iterative formula:
$$
\gamma_{k+1} = \gamma_{k} - \frac{G(\gamma^k)}{G^\prime(\gamma^k)}
$$

According to the property of sigmoid funcion, the gradient of $G(\gamma)$ is 
$$
G'(\gamma) = \sum_{x_i \in R_{mj}} \frac{d p(x_i, \gamma)}{d \gamma} = \sum_{x_i \in R_{mj}} p(x_i, \gamma)(1 - p(x_i, \gamma))
$$
So the iteration is
$$
\gamma_{k+1} = \gamma_k - \frac{\sum_{x_i \in R_{mj}}(p(x_i, \gamma_k) - y_i)}{\sum_{x_i \in R_{mj}} p(x_i, \gamma)(1 - p(x_i, \gamma))}
$$
it can be started from a random initial $\gamma_0$, for example using zero value. Don't forget that the optimal value shold be the root of $G(\gamma) = 0$. So excuting the iteration until the converged condition $|G(\gamma^*)| < \epsilon$ is satisfied, then $\gamma^*$ can be seemed as the final approximate solution. *Notice here is a numerical divergence problem in practice, when a pure True/False leaf $R_{mj}$ is presented.* In this situation, if the prediction $p(x_i, \gamma_k)$ is far from true label $y_i$, the ${G'}/{G}$ will divergence to infinity. As a result,  the probability $p(x_i, \gamma_{k+1})$ will not be calculated, because it needs the result of $e^{F + \gamma}$. But this situation often happens because it usually iterates hundreds or thousands in GradientBoosting algorithm, which means it generates hundread or thousands trees. So the pure leaf cannot be avoided even using shallow tree. A common method to overcome this problem is using a regularized factor i.e. $\gamma \leftarrow \gamma - \frac{G}{G' + \lambda}$ or $\gamma \leftarrow \gamma - \alpha G/G'$. Another alternative methods is keep the number of itearation equals to one in Newton updating process, but this leads to the need for more number of itarations in outer loop, i.e. more number of trees.

*results:*

<img src="/Users/eta/Library/Application Support/typora-user-images/image-20261018204822926.png" alt="image-20261018204822926" style="zoom:30%;" />

##### c. XGBoost

​	The XGBoost it's very inspired by GBDT, and it is very similar as GBDT. So in order not to confuse this two algorithms later, it's necessary  review the GBDT algorithm again before starting discussing the details of XGBoost. First of all, GBDT uses first-order gradient infotmation (residual $r_i$) to determine the structure of tree in boosting process. After that, it's trying to find the best leaf value for the regression tree with specific structure. Althought the leaf value is **the exact best value** theoritically, the GBDT uses numerical approaches to get a approximate value in practice, such as Newton-like methods. In summary, the GBDT uses the idea framework of boosting, but obtains the tree structure and leaf value separately in two steps. This is different from XGBoost, which unifies these two steps into a single framework, but uses a one-step approximation. Next we will describe how this approximation is introduced. 

​	Firstly, XGBoost also follows the idea of boosting, i.e the final value after $t$ iterations is 
$$
\hat{y}_i^{t} = \hat{y}_i^{(t-1)} + \eta f_{t}(x_i)
$$
where the $f_t$ is the regression tree in $t$th step (*notice that the convention of notation is slightly different from the previous text in order to be consistent with the notaion convention in the original paper*). The next two questions are how to determine the leaf values and structure of the tree $f_t$. Similarly, the starting point is the loss function, but XGBoost adds some regularity term to the loss function and rename it as **objective function**:

$$
\mathcal{L}^{(t)} = \sum_{i = 1}^n L\big(y_i, \sigma(\hat{y}_i^{(t-1)} + f_t(x_i))\big) + \sum_{k} \Omega(f_k)
$$

where the $\Omega$ can be seem as penalty term
$$
\Omega(f_k) = \gamma T + \frac{1}{2} \lambda \sum_j w_j^2
$$
the $T$ is the number of leaves of $f_k$ and $w_j$ is the leaf value of $f_k$. The $\gamma$ and $\lambda$ are two penalty parameters, the values of them are decided artificially in advance. It's not hard to see that the $\Omega$ term was added to decrese the complexity of tree and avoid numerical explosion of the leaf value. For binary classification problems similar to GBDT, if substituting cross entropy for loss $L$, it changes to 
$$
\mathcal{L}^{(t)} = \sum_{i = 1}^n L\big(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\big) + \sum_{k} \Omega(f_k)
$$
where 
$$
L\big(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\big) = \sum_{x_i \in R_{mj}} \Big[
-y_i (\hat{y}_i^{(t-1)} + f_t(x_i)) + \log(1 + e^{\hat{y}_i^{(t-1)} + f_t(x_i)})
\Big]
$$
Then XGBoost did a important change compared with GDBT—it didn't deal with objective function exactly but expand the objective function to second order using Taylor expansion. Notice that the loss is expanded at the point $\hat{y}_i$ because the $f_t(x_i)$ is small, this is the reason why above eqaution is not the function of $f_t(x_i)$ but $\hat{y}_i$ (compared with GDBT). Then the objective function is
$$
\mathcal{L}^{(t)} \simeq \sum_{i = 1}^n \Big[
L\big(y_i, \hat{y}_i^{(t-1)}\big) +
g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)
\Big] + \sum_{k} \Omega(f_k)
$$
here the $g_i$ and $h_i$ is the first-order gradient and second-order gradient respectively:
$$
g_i = \partial_{\hat{y}_i^{(t-1)}} L(y_i, \hat{y}_i^{(t-1)}) = p(\hat{y}_i^{(t-1)}) - y_i \\
h_i = \partial_{\hat{y}_i^{(t-1)}}^2 L(y_i, \hat{y}_i^{(t-1)}) = p(\hat{y}_i^{(t-1)}) \big(1 - p(\hat{y}_i^{(t-1)})\big)
$$
At $t$-th step the zero-order term $L(y_i, \hat{y}_x^{(t-1)})$ is constant term and the penal term $\Omega$ only left $\Omega(f_t)$, so the objective function can be further rewrited as 
$$
\tilde{\mathcal{L}}^{(t)} \simeq \sum_{i = 1}^n \Big[
g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)
\Big] + \gamma T + \frac{1}{2} \lambda \sum_j^T w_j^2 \\
\simeq \sum_j^T \Big[
(\sum_{i \in I_j} g_i) w_j + \frac{1}{2}(\sum_{i\in I_j}h_i + \lambda) w_j^2
\Big] + \gamma T
$$
where the $I_j = \{i | x_i \in \text{leaf} \ j\}$ and $f_t(x_i) = w_j,\ i \in I_j $. This is the most simplified objective function, it's a function of $w_j$ which is the parameters needed to be optimized. So we can get the optimal value $w_j^*$ according the *approximate* objective function
$$
w_j^* = - \frac{\sum_{i \in I_j}g_i}{\sum_{i \in I_j}h_i + \lambda}
$$
And subsitituting the $w_j^*$ into $\tilde{\mathcal{L}}^{(t)}$ can get the optimal objective function
$$
{\tilde{\mathcal{L}}^{(t)}}^* = -\frac{1}{2}\sum_j^T \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T
$$
Let's stop here temporarily and look back carefully. The XGBoost get a approximate solvable objective function using second-order taylor expansion, and it's easy to get the "optimal" tree weight and objective function, here the "optimal" is relative to the $\tilde{\mathcal{L}}^{(t)}$ rather than the $\mathcal{L}^{(t)}$. Then the XGBoost did a extremely important innovate compared with orther boosting methods—it trys to find not only optimal wight $w_j$ but also the optimal structure for $f_t$, which can be decided by the change of ${\tilde{\mathcal{L}}^{(t)}}^*$ instead of MSE or Gini impurity before and after splitting. So XGBoost defines the splitting difference of  ${\tilde{\mathcal{L}}^{(t)}}^*$ for a specific leaf $j$
$$
\Delta \mathcal{L} = {\tilde{\mathcal{L}}^{(t)}}^*_{\text{parent}} - {\tilde{\mathcal{L}}^{(t)}}^*_{\text{children}}
= \frac{1}{2}\Big[
\frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} + 
\frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} - 
\frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I_j} h_i + \lambda}
\Big] - \gamma \\
$$
where the 
$$
{\tilde{\mathcal{L}}^{(t)}}^*_{\text{parent}} = -\frac{1}{2} \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma \\
{\tilde{\mathcal{L}}^{(t)}}^*_{\text{children}} = {\tilde{\mathcal{L}}^{(t)}}^*_{\text{R}} + {\tilde{\mathcal{L}}^{(t)}}^*_{\text{L}} \\
{\tilde{\mathcal{L}}^{(t)}}^*_{\text{R}} = -\frac{1}{2} \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} + \gamma \\
{\tilde{\mathcal{L}}^{(t)}}^*_{\text{L}} = -\frac{1}{2} \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \gamma \\
$$
and the $I = I_R \cup I_L$, and here $T = T_L = T_R = 1$ because it's a process which splitting a parent leaf to two children leaf. Once we have this value to judge the validity for a alternative splitting, then it can be used to find a optimal structure for the tree $f_t(x_i)$, this is the key to why XGBoost works so well compared with other boosting methods. Keep reminding that this judgement is based on the second-order-approximate objective function, but most interesting thing is that this approximation is perfectly acceptabel because it's a **boosting** algorithm.
$$
F_{t-1} \rightarrow \hat{y}_i^{t-1} \rightarrow g_i\ \&\ h_i \rightarrow \text{Greedy split} \rightarrow \{w_j^*\} \rightarrow F_t \rightarrow \cdots
$$
*results:*

![image-20261111170428696](/Users/eta/Library/Application Support/typora-user-images/image-20261111170428696.png)

> Tips:
>
> + Initial prediction in Gradient Boosting:
>
>   $F_0$ is a constant baseline, not the first fitted tree. Before adding any trees, choose the constant that minimizes the total training loss. For binary labels $y_i \in \{0,1\}$ and cross-entropy loss, let $\bar{y}$ be the proportion of positive samples in the training set:
>
> $$
> F_0(x) = c^*, \qquad c^* = \operatorname{argmin}_c \sum_{i=1}^{N} L(y_i, \sigma(c))
> $$
>
> $$
> \frac{d}{dc}\sum_{i=1}^{N}L(y_i,\sigma(c)) = N\sigma(c)-\sum_{i=1}^{N}y_i = 0
> $$
>
> $$
> \sigma(c^*) = \bar{y}, \qquad F_0 = \log \frac{\bar{y}}{1-\bar{y}}
> $$
>
>   This initializes every sample with the same predicted probability $\bar{y}$; later trees learn corrections to its log-odds. The finite solution assumes $0<\bar{y}<1$. At the endpoints, the optimal logit tends to infinity in magnitude; clipping the probability avoids an infinite numerical initialization, although a training set containing both classes is needed for a meaningful binary classification task. This initialization applies to the binary logistic Gradient Boosting setup discussed above; the AdaBoost derivation instead starts with $F_0=0$ and uniform sample weights.
>
> + Early stopping:
>
>   Adding more trees can keep reducing training loss while eventually hurting performance on unseen data. Monitor a chosen metric on a separate validation set after each boosting round. If the metric does not improve for a specified number of rounds (the *patience*), stop training and use the ensemble from the best validation round. For example, with a patience of 20, if the best validation loss occurs at round 80 and no improvement occurs in rounds 81–100, training stops at round 100 and the selected ensemble contains the first 80 trees.
>
>   The learning rate $\eta$ and the number of trees should be considered together: a smaller $\eta$ makes each update more conservative and usually requires more trees. It does not guarantee better generalization, so select both using validation performance. Keep the test set separate from early stopping and hyperparameter selection. For Titanic, use a stratified split of the labeled training data, fit learned preprocessing only on the training portion, and use cross-validation when a single split gives unstable results.

### References

[1] Breiman, L. (1996). **Bagging Predictors**. *Machine Learning*, 24, 123–140. https://doi.org/10.1007/BF00058655

[2] Breiman, L. (2001). **Random Forests**. *Machine Learning*, 45, 5–32. [Original paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf).

[3] Freund, Y., & Schapire, R. E. (1997). **A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting**. *Journal of Computer and System Sciences*, 55(1), 119–139. https://doi.org/10.1006/jcss.1997.1504

[4] Friedman, J. H. (2001). **Greedy Function Approximation: A Gradient Boosting Machine**. *The Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

[5] Chen, T., & Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System**. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. [Original paper](https://arxiv.org/abs/1603.02754). https://doi.org/10.1145/2939672.2939785
