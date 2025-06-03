# Tiny Project: Linear System Implementation and Linear Regression

## Overview

This project, developed as part of Programming 2 (CS-2023), implements a foundational linear algebra system and applies it to construct a linear regression model for predicting relative CPU performance. The project is divided into two main components:

---

## Part A: Linear Algebra System

### A.1: Vector Class

- **Memory Management**
  - Constructors and destructors
- **Operator Overloading**
  - Assignment (`=`)
  - Unary minus (`-`)
  - Binary operators (`+`, `-`, `*` for vector-vector and vector-scalar operations)
- **Bounds Checking**
  - Overloaded square brackets (`[]`) for 0-based indexing
  - Overloaded round brackets (`()`) for 1-based indexing
- **Private Members**
  - `mSize`: Size of the array
  - `mData`: Pointer to data elements

---

### A.2: Matrix Class

- **Private Members**
  - `mNumRows`, `mNumCols`: Number of rows and columns
  - `mData`: Pointer to matrix data
- **Memory Management**
  - Constructor (rows, columns) initializes all entries to zero
  - Copy constructor
  - Destructor
- **Access Functions**
  - Public methods for retrieving the number of rows and columns
  - Overloaded round brackets (`()`) for 1-based indexing
- **Operator Overloading**
  - Assignment (`=`)
  - Unary minus (`-`)
  - Binary operators (`+`, `-`, `*` for matrix-matrix, matrix-vector, and matrix-scalar operations)
- **Other Matrix Properties**
  - Determinant (for square matrices)
  - Inverse (for square matrices)
  - Pseudo-inverse (Moore-Penrose inverse)

---

### A.3: Linear System Class

- **Private Members**
  - `mSize`: Size of the system
  - `mpA`: Pointer to matrix
  - `mpb`: Pointer to right-hand side vector
- **Memory Management**
  - Constructor (Matrix, Vector)
  - Copy constructor
  - Destructor
- **Solving Methods**
  - Gaussian elimination with pivoting
  - Virtual `Solve` method (for inheritance)
  - Symmetry check method
- **Derived Class: `PosSymLinSystem`**
  - Specialized solver for positive definite symmetric linear systems

---

### A.4: Non-Square Linear System Class

- **Solving Methods**
  - Moore-Penrose Inverse
  - Tikhonov Regularization

---

## Part B: Linear Regression for CPU Performance

### B.1: Dataset: Computer Hardware Dataset

- **Instances:** 209
- **Features (for each Instance):**
    1. Vendor name
    2. Model name
    3. MYCT: Machine cycle time (ns)
    4. MMIN: Minimum main memory (KB)
    5. MMAX: Maximum main memory (KB)
    6. CACH: Cache memory (KB)
    7. CHMIN: Minimum channels (units)
    8. CHMAX: Maximum channels (units)
    9. PRP: Published relative performance
    10. ERP: Estimated relative performance

---

### B.2: Linear Regression Model

The model predicts relative performance as follows:

```
PRP = x₁·MYCT + x₂·MMIN + x₃·MMAX + x₄·CACH + x₅·CHMIN + x₆·CHMAX
```

where `x₁` to `x₆` are model parameters determined using the linear algebra system.

---

### B.3: Training Process

- Dataset split: 80% training, 20% testing
- Evaluation metric: Root Mean Square Error (RMSE)

---

## Contributors

This project was developed by:

- Nguyễn Thùy Anh (Student ID: 10423198)
- Nguyễn Bảo Thành Đạt (Student ID: 10423136)
- Trần Anh Thư (Student ID: 10423173)
- Nguyễn Huỳnh Kim Thoa (Student ID: 10423172)
- Nguyễn Hồ Tuyết Phương (Student ID: 10423165)
- Trịnh Thị Như Bình (Student ID: 10423130)

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the conditions outlined in the license.
