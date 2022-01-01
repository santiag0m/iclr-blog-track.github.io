---
layout: post
title: Why Are Kronecker Products So Effective?
tags: [kronecker, quaternion, parameter efficient, tensor decomposition, SVD]
author: Anonymous
---

As soon as the [ICLR 2021's Outstading Paper Awards](https://iclr-conf.medium.com/announcing-iclr-2021-outstanding-paper-awards-9ae0514734ab) were announced, one paper immediately caught my attention from the list: "[Beyond Fully-Connected Layers with Quaternions: Parameterization of Hypercomplex Multiplications with $$1/n$$ Parameters](https://openreview.net/forum?id=rcQdycl0zyk)". Although the title might seem a bit daunting to those unfamiliar with quaternion algebra, the authors provide enough context to understand the core components of their approach and how they derive a new layer construction that generalizes the inductive biases behind quaternion-based Neural Networks, achieving sub-linear parameter scaling for Fully-Connected layers in Natural Language applications.

While it is not mentioned in the paper, it turns out that the effectiveness of the proposed layer scheme can also be explained using a much more widely known concept: the Singular Value Decomposition (SVD). In this blog post we will review the parametrization proposed by the authors, which links quaternion-based Neural Networks with the Kronecker Product, and later explain how the Kronecker Product provides a connection with a parallel line of research into parameter efficient Neural Networks for Computer Vision based on SVD.

### The PHM Layer

The main contribution of the paper is the "parameterized
hypercomplex multiplication (PHM) layer", a new layer that can replace fully connected (FC) layers with high parameter efficiency.

Instead of having a normal FC layer like this:

$$ \bf{y} = FC(\bf{x}) = \bf{W}x + b$$

We would have a PHM layer:

$$ \bf{y} = PHM(\bf{x}) = \bf{H}x + b$$

For both layers we are learning a linear mapping ($$\bf{W}$$ or $$\bf{H} \in \mathbb{R}^{k \times d}$$) of the input $$\bf{x}$$.

To have a clear understanding of what the proposed layer does, the authors introduce the Kronecker Product. For matrices $$\bf{A} \in \mathbb{R}^{m \times n}$$ and $$\bf{B} \in \mathbb{R}^{p \times q}$$, the Kronecker Product $$\otimes$$ is defined as:

$$
\begin{align*}
\bf{A} \otimes \bf{B} = \begin{bmatrix}
    a_{11}\bf{B} & \dots  & a_{1n}\bf{B} \\
    \vdots & \ddots & \vdots \\
    a_{m1}\bf{B} & \dots  & a_{mn}\bf{B}
    \end{bmatrix} \in \mathbb{R}^{mp \times nq}
\end{align*}
$$

The end result of applying the Kronecker Product to two matrices, is another matrix (a block matrix). With the assumption that both dimensions $$k$$ and $$d$$ are divisible by a user-selected positive integer $$n$$, the matrix $$\bf{H}$$ from the PHM layer can now be defined:

$$
\begin{align}
\bf{H} = \sum_{i=1}^n \bf{A_i} \otimes \bf{S_i}
\end{align}
$$

Where $$\bf{A_i} \in \mathbb{R}^{n \times n}$$ and $$\bf{S_i} \in \mathbb{R}^{\frac{k}{n} \times \frac{d}{n}}$$.

Such construction makes $$\bf{H}$$ very efficient in terms of parameter count, with approximately $$1/n$$ the number of parameters of an FC layer matrix $$\bf{W}$$. Assuming that $$kd > n^4$$, which is the case for high dimensional latent spaces found in practice.

One of the first things I questioned after seeing this equation was the restriction of $$\bf{A_i}$$ to be a square matrix. The authors provide an intuitive explanation from the point of view of quaternion multiplication (hence the name "hypercomplex multiplication", as the quaternion number system is a kind of [hypercomplex number system](https://en.wikipedia.org/wiki/Hypercomplex_number)). A nice property of quaternion multiplication (called the Hamilton Product), is that it can be rewritten as the following matrix:

$$
\begin{align}
\begin{bmatrix}
    Q_r & -Q_x & -Q_y & -Q_z \\
    Q_x & Q_r & -Q_z & Q_y \\
    Q_y & Q_z & Q_r & -Q_x \\
    Q_z & -Q_y & Q_x & Q_r \\
\end{bmatrix}
\begin{bmatrix}
    P_r \\
    P_x\\
    P_y\\
    P_z \\
\end{bmatrix},
\end{align}
$$

Where each subscript is associated with the quaternion unit basis.

This matrix can be interpreted as defining a rotation $$Q$$ of a 3-Dimensional vector $$P$$, which is very useful as an inductive bias to learn rotations inside Neural Networks (an experiment demonstrated in the paper). However, in its common form is not very useful for other dimensions, so the authors propose to reformulate it as the sum of Kronecker Products:

<div style="overflow-x: scroll">
$$
\begin{align}
\label{eq:ASQ_kron}
\underbrace{
\begin{bmatrix}
    1 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 \\
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1 \\
\end{bmatrix}
}_{\bf{A_1}}
\otimes
\underbrace{
\begin{bmatrix}
    Q_r \\
\end{bmatrix}
}_{\bf{S_1}}
+
\underbrace{
    \begin{bmatrix}
    0 & -1 & 0 & 0 \\
    1 & 0 & 0 & 0 \\
    0 & 0 & 0 & -1 \\
    0 & 0 & 1 & 0 \\
\end{bmatrix}
}_{\bf{A_2}}
\otimes
\underbrace{
\begin{bmatrix}
    Q_x \\
\end{bmatrix}
}_{\bf{S_2}}
+
\underbrace{
    \begin{bmatrix}
    0 & 0 & -1 & 0 \\
    0 & 0 & 0 & 1 \\
    1 & 0 & 0 & 0 \\
    0 & -1 & 0 & 0 \\
\end{bmatrix}
}_{\bf{A_3}}
\otimes
\underbrace{
\begin{bmatrix}
    Q_y \\
\end{bmatrix}
}_{\bf{S_3}}
+
\underbrace{
    \begin{bmatrix}
    0 & 0 & 0 & -1 \\
    0 & 0 & -1 & 0 \\
    0 & 1 & 0 & 0 \\
    1 & 0 & 0 & 0 \\
\end{bmatrix}
}_{\bf{A_4}}
\otimes
\underbrace{
\begin{bmatrix}
    Q_z \\
\end{bmatrix}
}_{\bf{S_4}}
.
\end{align}
$$
</div>

As can be seen, the matrices $$\bf{A_i} \in \mathbb{R}^{4 \times 4}$$ and $$\bf{S_i} \in \mathbb{R}^{\frac{4}{4} \times \frac{4}{4}}$$ are equivalent to the previous rotation matrix, which demonstrates that a PHM layer with $$n=4$$ can learn quaternion multiplication. Given that the same result holds for $$8D$$ (octonions), $$16D$$ (sedenions), and the fact that $$n$$ can take more values than just $$\{4, 8, 16\}$$, the PHM is said to generalize hypercomplex multiplication to $$nD$$.

To close this section about the PHM layer, I want to show the results they achieved applying the layer to machine translation, which offers great results in parameter efficiency without sacrificing much performance:

<p align="center">
  <img width="60%" src="{{ '/public/images/2021-12-01-kronecker-effective/phm_transformer_results.jpeg' | relative_url }}">
</p>

### Why are Kronecker Products effective then?

This paper caught my attention because I had read about the Kronecker Product being used in a similar manner for Convolutional Neural Networks. In particular, a 2015 paper called [**Exploiting Local Structures with the Kronecker Layer in Convolutional Networks**](https://arxiv.org/abs/1512.09194).

In this paper two new types of layers are proposed. First the Kronecker Fully-Connected (KFC) layer:

$$
\begin{align}
\mathbf{L_{i+1}} = f\left(\left(\sum_{i=1}^{r} \bf{A_i} \otimes \bf{B_i}\right)\bf{L_i} + \bf{b}_i\right),
\end{align}
$$

where $$\bf{A_i} \in \mathbb{R}^{m^{(i)}\times n^{(i)}}$$ and $$\bf{B_i} \in \mathbb{R}^{\frac{k}{m^{(i)}} \times \frac{d}{n^{(i)}}}$$.

And its generalization, the Kronecker Convolutional (KConv) layer, which approximates a convolutional kernel as follows:

$$
\begin{align}
\mathcal{W} \approx \sum_{i=1}^{r}\mathcal{A}_i\otimes\mathcal{B}_i,
\end{align}
$$

where $$\mathcal{A}_i$$ and $$\mathcal{B}_i$$ are $$4D$$ tensors with similar shape constraints as the cases before. Also note that there is no restriction for either $$\bf{A_i}$$ or $$\mathcal{A}_i$$ to be square, or for the number of summed products $$r$$ to be equal to $$n$$.

In contrast to PHM, the authors of the KConv paper arrive to the sum of Kronecker Products not by construction, but by improving on the ideas about low rank decomposition of Convolutional Neural Networks proposed in [**Speeding up convolutional Neural Networks with low rank expansions**](https://arxiv.org/abs/1405.3866) and [**Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation**](https://arxiv.org/abs/1404.0736), amongst others.

Particular emphasis must be made on their use of the duality between approximating weight matrices using the sum of Kronecker Products and SVD. This duality is demonstrated in Section 5.5 of the paper [**Approximation with Kronecker Products**](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.1924&rep=rep1&type=pdf), and it is crucial, as it gives a hint on why such parametrizations work in practice:

_"Next, we consider the situation when the matrix A to be approximated is a sum of Kronecker products:_

$$
A=\sum_{i=1}^{p}\left(G_{i} \otimes F_{i}\right) .
$$

_Assume that each $$G_i \in \mathbb{R}^{m1 \times n1}$$ and each $$F_i\in \mathbb{R}^{m2 \times n2}$$. It follows that if $$f_i = vec(Fi)$$ and $$g_i = vec(G_i)$$, then:_

$$
\tilde{A}=\mathcal{R}(A)=\sum_{i=1}^{p} \mathcal{R}\left(G_{i} \otimes F_{i}\right)=\sum_{i=1}^{p} g_{i} f_{i}^{T}
$$

_is a rank-$$p$$ matrix."_

While explaining the rearrangement operation $$\mathcal{R}(A)$$ is beyond the scope of this post (I highly encourage you to read the paper), this result shows how solving the problem of approximating a matrix $$A$$ with the sum of $$p$$ Kronecker Products is equivalent to the rank-$$p$$ SVD of a rearranged version of $$A$$.

As it is the case with many signals in the real world, the intrinsic dimensionality of the transformer weights in the PHM paper is likely to be small. As such, low-rank approximations might be able to capture most of the model behavior with few parameters, explaining the efficiency of the Kronecker Product approach.

### Final thoughts

I first came across the Kronecker Product back in 2018, when I worked on a university course presentation about incorporating large scale context in Neural Networks. Although I started by looking at what was novel at the time ([**Deformable CNNs**](https://arxiv.org/abs/1703.06211) and the Atrous Spatial Pyramid Pooling scheme in [**Deeplab**](https://arxiv.org/abs/1706.05587)), it wasn't until I found a great blog post by Ferenc Huszár outlining the relationship between [Dilated Convolutions and Kronecker Factored Convolutions](https://www.inference.vc/dilated-convolutions-and-kronecker-factorisation/) that I became captivated by the subject.

As it turns out, there has been a long chain of papers on making Neural Networks efficient, all with different takes on which is the best way to do Matrix (or Tensor) Decomposition. From the ideas that influenced the development of the KConv layers to the novel connection with hypercomplex multiplication proposed with the PHM layer, I have become convinced that the Kronecker Product, and the inductive biases it can encode, will be a crucial tool in the path to understanding Neural Networks.

## References

- Zhang, Aston, et al. "Beyond Fully-Connected Layers with Quaternions: Parameterization of Hypercomplex Multiplications with $$1/n$$ Parameters." arXiv preprint [arXiv:2102.08597](https://openreview.net/forum?id=rcQdycl0zyk) (2021).
- Zhou, Shuchang, et al. "Exploiting local structures with the kronecker layer in convolutional networks." arXiv preprint [arXiv:1512.09194](https://arxiv.org/abs/1512.09194) (2015).
- Jaderberg, Max, Andrea Vedaldi, and Andrew Zisserman. "Speeding up convolutional Neural Networks with low rank expansions." arXiv preprint [arXiv:1405.3866](https://arxiv.org/abs/1405.3866) (2014).
- Denton, Emily, et al. "Exploiting linear structure within convolutional networks for efficient evaluation." arXiv preprint [arXiv:1404.0736](https://arxiv.org/abs/1404.0736) (2014).
- Van Loan, Charles F., and Nikos Pitsianis. ["Approximation with Kronecker products."](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.1924&rep=rep1&type=pdf) Linear algebra for large scale and real-time applications. Springer, Dordrecht, 1993. 293-314.
- Dai, Jifeng, et al. ["Deformable convolutional networks."](https://arxiv.org/abs/1703.06211) Proceedings of the IEEE international conference on computer vision. 2017.
- Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation." arXiv preprint [arXiv:1706.05587](https://arxiv.org/abs/1706.05587) (2017).
- "Dilated Convolutions and Kronecker Factored Convolutions." [https://www.inference.vc/dilated-convolutions-and-kronecker-factorisation/](https://www.inference.vc/dilated-convolutions-and-kronecker-factorisation/)
