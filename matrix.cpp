#include "vector.h"
#include "matrix.h"
#include "linearSystem.h"

    // Constructor
    Matrix::Matrix(int numRows, int numCols) {
        assert(numRows > 0 && numCols > 0);
        mNumRows = numRows;
        mNumCols = numCols;

    mData = new double*[mNumRows];
    for (int i = 0; i < mNumRows; ++i) {
        mData[i] = new double[mNumCols];
        for (int j = 0; j < mNumCols; ++j) {
            mData[i][j] = 0.0;
        }
    }
}

// Copy constructor
Matrix::Matrix(const Matrix& other) {
    mNumRows = other.mNumRows;
    mNumCols = other.mNumCols;

    mData = new double*[mNumRows];
    for (int i = 0; i < mNumRows; ++i) {
        mData[i] = new double[mNumCols];
        for (int j = 0; j < mNumCols; ++j) {
            mData[i][j] = other.mData[i][j];
        }
    }
}

// Static method to create an identity matrix
Matrix Matrix::IdentityMatrix(int size) {
    Matrix I(size, size);
    for (int i = 1; i <= size; ++i) {
        I(i, i) = 1.0; // Diagonal entries set to 1.0
    }
    return I;
}

// Compute the pseudo-inverse using the Moore-Penrose method
Matrix Matrix::PseudoInverse() const { 
    if (mNumRows >= mNumCols) {
        // Over-determined: (A^T A)^-1 A^T
        Matrix A_T = this->Transpose();
        Matrix ATA = A_T * (*this);

<<<<<<< HEAD
            // Invert ATA
            Matrix ATA_inv(mNumCols, mNumCols);
            for (int i = 1; i <= mNumCols; ++i) {
                Vector e(mNumCols);
                e(i) = 1.0;
                LinearSystem ls(ATA, e);
                Vector col = ls.Solve();
                for (int j = 1; j <= mNumCols; ++j) {
                    ATA_inv(j, i) = col(j);
                }
            }
            return ATA_inv * A_T;
        } else {
            // Under-determined: A^T (A A^T)^-1
            Matrix A_T = this->Transpose();
            Matrix AAT = (*this) * A_T;

            // Invert AAT
            Matrix AAT_inv(mNumRows, mNumRows);
            for (int i = 1; i <= mNumRows; ++i) {
                Vector e(mNumRows);
                e(i) = 1.0;
                LinearSystem ls(AAT, e);
                Vector col = ls.Solve();
                for (int j = 1; j <= mNumRows; ++j) {
                    AAT_inv(j, i) = col(j);
                }
            }
            // *** FIX: Remove .Transpose() ***
            return A_T * AAT_inv;
        }
    }

    // Transpose the matrix
    Matrix Matrix::Transpose() const { 
        Matrix result(mNumCols, mNumRows); // Swap rows and columns
        for (int i = 1; i <= mNumRows; ++i) {
=======
        // Invert ATA
        Matrix ATA_inv(mNumCols, mNumCols);
        for (int i = 1; i <= mNumCols; ++i) {
            Vector e(mNumCols);
            e(i) = 1.0;
            LinearSystem ls(ATA, e);
            Vector col = ls.Solve();
>>>>>>> b6bd0ce7994bdd6ab87a60d6688d44c75335c6f3
            for (int j = 1; j <= mNumCols; ++j) {
                ATA_inv(j, i) = col(j);
            }
        }
        return ATA_inv * A_T;
    } else {
        // Under-determined: A^T (A A^T)^-1
        Matrix A_T = this->Transpose();
        Matrix AAT = (*this) * A_T;

        // Invert AAT
        Matrix AAT_inv(mNumRows, mNumRows);
        for (int i = 1; i <= mNumRows; ++i) {
            Vector e(mNumRows);
            e(i) = 1.0;
            LinearSystem ls(AAT, e);
            Vector col = ls.Solve();
            for (int j = 1; j <= mNumRows; ++j) {
                AAT_inv(j, i) = col(j);
            }
        }
        return (A_T * AAT_inv).Transpose();
    }
}

// Transpose the matrix
Matrix Matrix::Transpose() const { 
    Matrix result(mNumCols, mNumRows); // Swap rows and columns
    for (int i = 1; i <= mNumRows; ++i) {
        for (int j = 1; j <= mNumCols; ++j) {
            result(j, i) = (*this)(i, j); // Use 1-based indexing
        }
    }
    return result;
}

// Destructor
Matrix::~Matrix() {
    for (int i = 0; i < mNumRows; ++i) {
        delete[] mData[i];
    }
    delete[] mData;
}

// Assignment
Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) return *this;

    // Clean up
    for (int i = 0; i < mNumRows; ++i)
        delete[] mData[i];
    delete[] mData;

    // Copy
    mNumRows = other.mNumRows;
    mNumCols = other.mNumCols;
    mData = new double*[mNumRows];
    for (int i = 0; i < mNumRows; ++i) {
        mData[i] = new double[mNumCols];
        for (int j = 0; j < mNumCols; ++j) {
            mData[i][j] = other.mData[i][j];
        }
    }

    return *this;
}

// Getters
int Matrix::GetNumRows() const { return mNumRows; }
int Matrix::GetNumCols() const { return mNumCols; }

    // Access operator ()
    double& Matrix::operator()(int i, int j) {
        assert(i >= 1 && i <= mNumRows && j >= 1 && j <= mNumCols);
        return mData[i - 1][j - 1];
    }

    // Const access operator ()
    const double& Matrix::operator()(int i, int j) const {
        assert(i >= 1 && i <= mNumRows && j >= 1 && j <= mNumCols);
        return mData[i - 1][j - 1];
    }

    // + operator
    Matrix Matrix::operator+(const Matrix& other) const {
        assert(mNumRows == other.mNumRows && mNumCols == other.mNumCols);
        Matrix result(mNumRows, mNumCols);
        for (int i = 0; i < mNumRows; ++i)
            for (int j = 0; j < mNumCols; ++j)
                result.mData[i][j] = mData[i][j] + other.mData[i][j];
        return result;
    }

    // - operator
    Matrix Matrix::operator-(const Matrix& other) const {
        assert(mNumRows == other.mNumRows && mNumCols == other.mNumCols);
        Matrix result(mNumRows, mNumCols);
        for (int i = 0; i < mNumRows; ++i)
            for (int j = 0; j < mNumCols; ++j)
                result.mData[i][j] = mData[i][j] - other.mData[i][j];
        return result;
    }

// * scalar
Matrix Matrix::operator*(double scalar) const {
    Matrix result(mNumRows, mNumCols);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[i][j] = mData[i][j] * scalar;
    return result;
}

<<<<<<< HEAD
    // * vector
    Vector Matrix::operator*(const Vector& vec) const {
    assert(mNumCols == vec.getSize());
    Vector result(mNumRows);
    for (int i = 1; i <= mNumRows; ++i) {
        result(i) = 0.0;
        for (int j = 1; j <= mNumCols; ++j)
            result(i) += (*this)(i, j) * vec(j);
    }
    return result;
=======
// * vector
Vector Matrix::operator*(const Vector& vec) const {
    assert(mNumCols == vec.getSize()); 
    Vector result(mNumRows);
    for (int i = 0; i < mNumRows; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < mNumCols; ++j)
            result[i] += mData[i][j] * vec[j];
>>>>>>> b6bd0ce7994bdd6ab87a60d6688d44c75335c6f3
    }
    return result;
}

    // * matrix
    Matrix Matrix::operator*(const Matrix& other) const {
        assert(mNumCols == other.mNumRows);
        Matrix result(mNumRows, other.mNumCols);
        for (int i = 0; i < mNumRows; ++i) {
            for (int j = 0; j < other.mNumCols; ++j) {
                result.mData[i][j] = 0.0;
                for (int k = 0; k < mNumCols; ++k)
                    result.mData[i][j] += mData[i][k] * other.mData[k][j];
            }
        }
        return result;
    }

// Unary minus operator
Matrix Matrix::operator-() const {
    Matrix result(mNumRows, mNumCols);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[i][j] = -mData[i][j];
    return result;
}

// Print for debugging
void Matrix::Print() const {
    for (int i = 0; i < mNumRows; ++i) {
        std::cout << "[ ";
        for (int j = 0; j < mNumCols; ++j) {
            std::cout << mData[i][j] << " ";
        }
        std::cout << "]\n";
    }
}

    double Matrix::Determinant() const {
        assert(mNumRows == mNumCols);
        int n = mNumRows;
        if (n == 1) return mData[0][0];
        if (n == 2) return mData[0][0] * mData[1][1] - mData[0][1] * mData[1][0];
        if (n == 3) {
            return mData[0][0] * (mData[1][1] * mData[2][2] - mData[1][2] * mData[2][1])
                 - mData[0][1] * (mData[1][0] * mData[2][2] - mData[1][2] * mData[2][0])
                 + mData[0][2] * (mData[1][0] * mData[2][1] - mData[1][1] * mData[2][0]);
        }

    double det = 0.0;
    for (int p = 0; p < n; ++p) {
        Matrix subMat(n - 1, n - 1);
        for (int i = 1; i < n; ++i) {
            int colIdx = 0;
            for (int j = 0; j < n; ++j) {
                if (j == p) continue;
                subMat.mData[i - 1][colIdx] = mData[i][j];
                ++colIdx;
            }
        }
        det += ((p % 2 == 0) ? 1 : -1) * mData[0][p] * subMat.Determinant();
    }
    return det;
}

    Matrix Matrix::Inverse() const {
        assert(mNumRows == mNumCols);
        int n = mNumRows;
<<<<<<< HEAD
        Matrix A(*this);
        Matrix inv = Matrix::IdentityMatrix(n);

        for (int i = 1; i <= n; ++i) {
            // Find the pivot
            double pivot = A(i, i);
            if (std::abs(pivot) < 1e-12) {
                // Try to swap with a lower row
                bool swapped = false;
                for (int k = i + 1; k <= n; ++k) {
                    if (std::abs(A(k, i)) > 1e-12) {
                        // Swap rows i and k in both A and inv
                        for (int j = 1; j <= n; ++j) {
                            std::swap(A(i, j), A(k, j));
                            std::swap(inv(i, j), inv(k, j));
                        }
                        pivot = A(i, i);
                        swapped = true;
                        break;
                    }
                }
                if (!swapped) throw std::runtime_error("Matrix is singular or nearly singular");
            }
            // Normalize the pivot row
            for (int j = 1; j <= n; ++j) {
                A(i, j) /= pivot;
                inv(i, j) /= pivot;
            }
            // Eliminate other rows
            for (int k = 1; k <= n; ++k) {
                if (k == i) continue;
                double factor = A(k, i);
                for (int j = 1; j <= n; ++j) {
                    A(k, j) -= factor * A(i, j);
                    inv(k, j) -= factor * inv(i, j);
                }
            }
        }
=======
        double det = this->Determinant();
        const double EPS = 1e-12;
        assert(std::abs(det) > EPS && "Matrix is singular or nearly singular");

    Matrix inv(n, n);
    if (n == 1) {
        inv.mData[0][0] = 1.0 / mData[0][0];
>>>>>>> b6bd0ce7994bdd6ab87a60d6688d44c75335c6f3
        return inv;
    }
    if (n == 2) {
        inv.mData[0][0] =  mData[1][1] / det;
        inv.mData[0][1] = -mData[0][1] / det;
        inv.mData[1][0] = -mData[1][0] / det;
        inv.mData[1][1] =  mData[0][0] / det;
        return inv;
    }
    // For n > 2, use cofactor expansion
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Matrix minor(n - 1, n - 1);
            int rowIdx = 0;
            for (int r = 0; r < n; ++r) {
                if (r == i) continue;
                int colIdx = 0;
                for (int c = 0; c < n; ++c) {
                    if (c == j) continue;
                    minor.mData[rowIdx][colIdx] = mData[r][c];
                    ++colIdx;
                }
                ++rowIdx;
            }
            double cofactor = ((i + j) % 2 == 0 ? 1 : -1) * minor.Determinant();
            inv.mData[j][i] = cofactor / det; // Transpose for adjugate
        }
    }
    return inv;
}

bool Matrix::Symmetric() const {
if (!Square()) return false;
for (int i = 0; i < mNumRows; i++) {
    for (int j = 0; j < i; j++) {
        if (mData[i][j] != mData[j][i]) {
            return false;
        }
    }
}
return true;
}
