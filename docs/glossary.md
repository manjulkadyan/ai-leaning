# Glossary

Plain-language definitions of the terms the course uses, each with the lesson that teaches it. It covers all of Month 1 and grows as lessons are added. Use your browser's find (Ctrl/Cmd + F) to jump to a word.

## A–C

- **Activation function.** A simple non-linear function applied to each neuron's output, such as ReLU. Without it, stacked layers collapse into one big matrix multiply. *[Day 12](lessons/phase-1/day-12-layer-is-matmul.md)*
- **Array (`ndarray`).** NumPy's grid of numbers that all share one type, stored in one contiguous block of memory. The basic data structure of ML code. *[Day 4](lessons/phase-1/day-04-numpy-arrays.md)*
- **Axis.** One dimension of an array. `axis=0` runs down the rows, `axis=1` across the columns; `a.sum(axis=0)` collapses the rows. *[Day 4](lessons/phase-1/day-04-numpy-arrays.md)*
- **Basis.** A set of independent vectors that spans a space, so every point has exactly one recipe (its coordinates) in terms of them. *[Day 23](lessons/phase-1/day-23-rank-span.md)*
- **Batch.** Several inputs processed together, stacked along a new first axis. A batch of 32 images of 224×224 RGB is a `(32, 224, 224, 3)` array. *[Day 12](lessons/phase-1/day-12-layer-is-matmul.md)*
- **Bias.** The number added after a neuron's weighted sum (\(\mathbf{w} \cdot \mathbf{x} + b\)). It lets the neuron shift its output up or down. *[Day 5](lessons/phase-1/day-05-vectors.md)*
- **Boolean mask.** An array of `True`/`False` values used to pick elements or rows, as in `apps[apps["rating"] > 4.5]`. *[Day 4](lessons/phase-1/day-04-numpy-arrays.md), [Day 16](lessons/phase-1/day-16-pandas-selecting.md)*
- **Broadcasting.** NumPy's rule for combining arrays of different shapes by lining them up from the right and stretching size-1 dimensions, without copying data. *[Day 9](lessons/phase-1/day-09-broadcasting.md)*
- **Column space.** Every output \(A\mathbf{x}\) a matrix can produce: the span of its columns. Its dimension is the rank. *[Day 23](lessons/phase-1/day-23-rank-span.md)*
- **Column vector / row vector.** A vector written as an \(n \times 1\) matrix (a column) or a \(1 \times n\) matrix (a row). In NumPy a plain 1-D array is neither until you reshape it. *[Day 10](lessons/phase-1/day-10-matrices-matmul.md)*
- **Copy-on-Write.** The pandas 3 rule that every selection behaves like a new object, so modifying it never changes the original. Chained assignment like `df[mask]["col"] = 0` therefore does nothing. *[Day 17](lessons/phase-1/day-17-pandas-cleaning.md)*
- **Correlation.** How strongly two columns move together, from −1 to 1; it is the cosine similarity of the two centred columns. *[Day 20](lessons/phase-1/day-20-week3-review.md)*
- **Cosine similarity.** How closely two vectors point the same way, from −1 (opposite) to 1 (same direction), ignoring their lengths. The standard score for comparing embeddings. *[Day 5](lessons/phase-1/day-05-vectors.md)*
- **Covariance matrix.** A square table of how much each pair of features varies together. Its eigenvectors are the directions PCA keeps. *[Day 24](lessons/phase-1/day-24-eigenvectors.md), [Day 25](lessons/phase-1/day-25-projections-pca.md)*
- **Curse of dimensionality.** In many dimensions the nearest and farthest points end up almost equally far away, so distance-based methods weaken. *[Day 22](lessons/phase-1/day-22-norms-distances.md)*

## D–F

- **DataFrame.** pandas' table: named columns (each a Series) that share one row index. Think of a Room table loaded into memory. *[Day 15](lessons/phase-1/day-15-pandas-dataframes.md)*
- **Determinant.** One number per square matrix telling how much it scales area (or volume). Zero means the matrix squashes space flat and has no inverse. *[Day 11](lessons/phase-1/day-11-transpose-identity-inverse.md)*
- **Dot product.** Multiply two vectors element by element and add up the results. Large when they point the same way; zero when they are perpendicular. *[Day 5](lessons/phase-1/day-05-vectors.md)*
- **dtype.** The element type of an array or column, such as `float32`, `int64` or pandas' `str`. It fixes memory size and which operations are allowed. *[Day 4](lessons/phase-1/day-04-numpy-arrays.md), [Day 15](lessons/phase-1/day-15-pandas-dataframes.md)*
- **Eigenvector / eigenvalue.** A direction a matrix only stretches, never turns, and the factor it stretches by: \(A\mathbf{v} = \lambda\mathbf{v}\). *[Day 24](lessons/phase-1/day-24-eigenvectors.md)*
- **Embedding.** A vector of numbers that represents a word, sentence, image or user so that similar things get nearby vectors. Introduced in Phase 4; the counting approach it replaces is on [Day 6](lessons/phase-1/day-06-week1-review.md).
- **Feature.** One measurable input column, such as an app's size or rating. A row of features is one example's input vector. *[Day 5](lessons/phase-1/day-05-vectors.md)*
- **FLOP.** One floating-point operation (an add or a multiply). A matrix multiply of \((m, n) @ (n, p)\) costs about \(2mnp\) FLOPs, which is how you estimate a model's speed. *[Day 12](lessons/phase-1/day-12-layer-is-matmul.md), [Day 13](lessons/phase-1/day-13-week2-review.md)*
- **Forward pass.** Running an input through a model's layers, in order, to get its output. *[Day 12](lessons/phase-1/day-12-layer-is-matmul.md)*

## G–L

- **groupby (split-apply-combine).** Split rows into groups by a key, compute something per group, and combine the results into one table. SQL's `GROUP BY`. *[Day 18](lessons/phase-1/day-18-pandas-groupby-merge.md)*
- **Identity matrix (\(I\)).** The square matrix with 1s on the diagonal and 0s elsewhere. Multiplying by it changes nothing, like multiplying a number by 1. *[Day 11](lessons/phase-1/day-11-transpose-identity-inverse.md)*
- **Index (pandas).** The row labels of a Series or DataFrame. Operations line up rows by index label, not by position. *[Day 15](lessons/phase-1/day-15-pandas-dataframes.md)*
- **Inverse (\(A^{-1}\)).** The matrix that undoes \(A\): \(A^{-1}A = I\). Only exists when the determinant isn't zero, and in practice you solve systems with `np.linalg.solve` instead of computing it. *[Day 11](lessons/phase-1/day-11-transpose-identity-inverse.md)*
- **Join / merge.** Combining two tables by matching key columns. An inner join keeps only matches; a left join keeps every row of the left table. *[Day 18](lessons/phase-1/day-18-pandas-groupby-merge.md)*
- **Jupyter notebook (`.ipynb`).** A document of code cells and text cells you run one at a time, with outputs saved under each cell. Every lesson offers its code as one. *[Day 1](lessons/phase-1/day-01-setup.md)*
- **L1 / L2 / L∞ norm.** Three ways to measure length: the sum of absolute values (Manhattan), the square root of the sum of squares (straight line), and the largest absolute value. *[Day 22](lessons/phase-1/day-22-norms-distances.md)*
- **Layer (linear / dense / fully connected).** Many neurons applied at once: \(\mathbf{y} = W\mathbf{x} + \mathbf{b}\). PyTorch calls it `nn.Linear`; LiteRT calls it `FULLY_CONNECTED`. *[Day 12](lessons/phase-1/day-12-layer-is-matmul.md)*
- **Least squares.** Fitting by making the sum of squared errors as small as possible; geometrically, projecting the data onto the column space. *[Day 25](lessons/phase-1/day-25-projections-pca.md)*
- **Linear independence.** No vector in the set can be built from the others. Dependent columns make a matrix lose rank. *[Day 23](lessons/phase-1/day-23-rank-span.md)*
- **LoRA (low-rank adaptation).** Fine-tuning a big model by freezing its weights \(W\) and learning a small low-rank update \(BA\), often under 1% of the parameters. *[Day 23](lessons/phase-1/day-23-rank-span.md)*

## M–O

- **Matrix.** A 2-D grid of numbers with shape (rows, columns). It can store a dataset (one row per example) or a transformation (a layer's weights). *[Day 10](lessons/phase-1/day-10-matrices-matmul.md)*
- **Matrix multiplication (`@`).** Each output entry is the dot product of a row of the left matrix with a column of the right one. Shapes must agree: \((m, n) @ (n, p) \to (m, p)\). *[Day 10](lessons/phase-1/day-10-matrices-matmul.md)*
- **NaN.** "Not a Number", the float value pandas and NumPy use for a missing measurement. Any arithmetic with NaN gives NaN, and `NaN == NaN` is `False`. *[Day 17](lessons/phase-1/day-17-pandas-cleaning.md)*
- **Nearest-neighbour search.** Finding the stored vectors closest to a query, the core of semantic search and recommendation. *[Day 22](lessons/phase-1/day-22-norms-distances.md)*
- **Neuron.** The smallest unit of a neural network: a dot product of inputs with weights, plus a bias, usually followed by an activation. *[Day 5](lessons/phase-1/day-05-vectors.md)*
- **NHWC / NCHW.** Two memory layouts for a batch of images: batch, height, width, channels (LiteRT/TFLite) or batch, channels, height, width (PyTorch). Feeding the wrong one gives garbage, not an error. *[Day 8](lessons/phase-1/day-08-shapes-indexing.md)*
- **Norm (\(\lVert \mathbf{x} \rVert\)).** The length of a vector. The usual one (L2) is the square root of the sum of squares. *[Day 5](lessons/phase-1/day-05-vectors.md)*

## P–R

- **Parameters (weights).** The numbers a model learns during training: every entry of every weight matrix and bias vector. A layer with \(n\) inputs and \(m\) outputs has \(mn + m\). *[Day 12](lessons/phase-1/day-12-layer-is-matmul.md)*
- **PCA (principal component analysis).** Rotating data onto the directions of largest variance (the covariance eigenvectors) so you can keep the few that matter. *[Day 25](lessons/phase-1/day-25-projections-pca.md)*
- **Permutation (shuffle) test.** Shuffling group labels many times to see how often chance alone produces a gap as big as the real one. *[Day 20](lessons/phase-1/day-20-week3-review.md)*
- **Pivot table.** A summary table that turns the values of one column into new columns, aggregating as it goes, like a spreadsheet pivot. *[Day 18](lessons/phase-1/day-18-pandas-groupby-merge.md)*
- **Power iteration.** Multiplying a vector by a matrix again and again (rescaling each time) until it lines up with the biggest eigenvector; the idea behind PageRank. *[Day 24](lessons/phase-1/day-24-eigenvectors.md)*
- **Projection.** The "shadow" of a vector on a line or plane: the closest point there, with the leftover perpendicular to it. *[Day 25](lessons/phase-1/day-25-projections-pca.md)*
- **Quantization (int8).** Storing weights and activations as 8-bit integers plus a scale factor instead of 32-bit floats: 4× smaller and much faster on phone NPUs, at a small accuracy cost. *[Day 13](lessons/phase-1/day-13-week2-review.md)*
- **Rank.** The number of independent columns (equivalently rows) of a matrix: how many dimensions its outputs really fill. *[Day 23](lessons/phase-1/day-23-rank-span.md)*
- **ReLU.** The activation \(\max(0, x)\): negative values become 0, positive values pass through. *[Day 12](lessons/phase-1/day-12-layer-is-matmul.md)*
- **Reshape.** Viewing the same numbers with a different shape, such as a 28×28 image as a 784-vector. The total number of elements must stay the same. *[Day 8](lessons/phase-1/day-08-shapes-indexing.md)*

## S–Z

- **Scalar.** A single number, as opposed to a vector or matrix. *[Day 5](lessons/phase-1/day-05-vectors.md)*
- **Series.** pandas' labelled 1-D array: values plus an index. One column of a DataFrame. *[Day 15](lessons/phase-1/day-15-pandas-dataframes.md)*
- **Shape.** The size of an array along each axis, as a tuple: `(3, 4)` means 3 rows and 4 columns. Most ML bugs are shape bugs. *[Day 4](lessons/phase-1/day-04-numpy-arrays.md), [Day 8](lessons/phase-1/day-08-shapes-indexing.md)*
- **Singular matrix.** A square matrix with determinant 0. It collapses some direction to nothing, so it has no inverse and its system of equations has no unique solution. *[Day 11](lessons/phase-1/day-11-transpose-identity-inverse.md)*
- **Span.** Every vector you can reach by scaling and adding a set of vectors. *[Day 23](lessons/phase-1/day-23-rank-span.md)*
- **Standardization.** Rescaling each feature to mean 0 and standard deviation 1, using statistics from the training set only, so no feature dominates distances. *[Day 22](lessons/phase-1/day-22-norms-distances.md)*
- **SVD (singular value decomposition).** Splitting any matrix into rotation × stretch × rotation; what libraries use to compute PCA and low-rank approximations. *[Day 25](lessons/phase-1/day-25-projections-pca.md)*
- **Tensor.** The general name for an array with any number of dimensions: a scalar is 0-D, a vector 1-D, a matrix 2-D, a batch of images 4-D. PyTorch and LiteRT call their arrays tensors. *[Day 4](lessons/phase-1/day-04-numpy-arrays.md)*
- **TF-IDF.** A way to weight word counts so that words common to every document (like "the") count less than distinctive ones. *[Day 6](lessons/phase-1/day-06-week1-review.md)*
- **Transpose (\(A^\top\)).** Flipping a matrix over its diagonal so rows become columns: shape \((m, n)\) becomes \((n, m)\). *[Day 11](lessons/phase-1/day-11-transpose-identity-inverse.md)*
- **Vector.** An ordered list of numbers, which you can picture as an arrow or a point in space. Everything in ML (an image, a sentence, a user) becomes one. *[Day 5](lessons/phase-1/day-05-vectors.md)*
- **Vectorization.** Replacing a Python loop with one array operation that runs in compiled code, usually 10–100× faster. *[Day 4](lessons/phase-1/day-04-numpy-arrays.md)*
- **View vs copy.** A view is a new array object that shares memory with the original (slicing, `reshape`); changing it changes the original. A copy has its own memory. *[Day 8](lessons/phase-1/day-08-shapes-indexing.md)*
- **Virtual environment (venv).** An isolated folder of Python packages for one project, so projects don't fight over versions. Like a per-module Gradle dependency set. *[Day 1](lessons/phase-1/day-01-setup.md)*
