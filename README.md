# Shift-MIL Pooling and Application
Shift MIL pooling layer based on Noise-And for Classification and Segmentation. It is the implementation of Shift-MIL pooling function based on "Surface Defect Segmentation with Multi-Column patch-wise U-net" by Zihao et al.

## Mathematics Equation
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
\begin{equation}\label{eq:S-MIL}
P^{c}=g^{c}\left(\left\{p_{j}^{c}\right\}\right)=\frac{\sigma\left(a\left(\left(p_{\bar{j}}^{c}\right)^{s}-b^{c}\right)\right)-\sigma\left(-a b^{c}\right)}{\sigma\left(a\left(1-b^{c}\right)\right)-\sigma\left(-a b^{c}\right)},
\end{equation}


