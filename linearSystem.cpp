#include "linearSystem.h"
#include "vector.h"
#include "matrix.h"

#include <cassert>
#include <cmath>
#include <algorithm>

LinearSystem::LinearSystem(const Matrix& A, const Vector& b) {
    assert(A.GetNumRows() == A.GetNumCols());
    assert(A.GetNumRows() == b.getSize());

    mSize = A.GetNumRows();
    mpA = new Matrix(A);
    mpb = new Vector(b);
}

LinearSystem::~LinearSystem() {
    delete mpA;
    delete mpb;
}

Vector LinearSystem::Solve() {
    int n = mSize;
    Matrix A(*mpA);
    Vector b(*mpb);

    // Forward elimination with partial pivoting
    for (int k = 0; k < n - 1; ++k) {
        // Pivoting
        int pivot = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(A(i + 1, k + 1)) > std::abs(A(pivot + 1, k + 1))) {
                pivot = i;
            }
        }
        if (pivot != k) {
            for (int j = 1; j <= n; ++j) std::swap(A(k + 1, j), A(pivot + 1, j));
            std::swap(b(k + 1), b(pivot + 1));
        }

        // Elimination
        for (int i = k + 1; i < n; ++i) {
            double factor = A(i + 1, k + 1) / A(k + 1, k + 1);
            for (int j = k + 1; j <= n; ++j) {
                A(i + 1, j) -= factor * A(k + 1, j);
            }
            b(i + 1) -= factor * b(k + 1);
        }
    }

    // Back substitution
    Vector x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = b(i + 1);
        for (int j = i + 1; j < n; ++j) {
            sum -= A(i + 1, j + 1) * x(j + 1);
        }
        x(i + 1) = sum / A(i + 1, i + 1);
    }

    return x;
}

// PosSymLinSystem implementation

PosSymLinSystem::PosSymLinSystem(Matrix& A, Vector& b) : LinearSystem(A, b) {
    // Check that the matrix is symmetric
    if (!A.Symmetric()) {
        throw std::invalid_argument("Matrix must be symmetric for PosSymLinSystem");
    }
}

Vector PosSymLinSystem::Solve() {
    // Conjugate Gradient method implementation
    const Matrix& A = *mpA;
    const Vector& b = *mpb;
    
    const int maxIterations = 1000;
    const double tolerance = 1e-10;
    
    Vector x(mSize); // Initial guess (all zeros)
    Vector r = b - MatrixVectorMultiply(A, x);
    Vector p = r;
    double rsold = DotProduct(r, r);
    
    for (int i = 0; i < maxIterations; i++) {
        Vector Ap = MatrixVectorMultiply(A, p);
        double alpha = rsold / DotProduct(p, Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        
        double rsnew = DotProduct(r, r);
        if (sqrt(rsnew) < tolerance) {
            break;
        }
        
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    }
    
    return x;
}

double PosSymLinSystem::DotProduct(const Vector& a, const Vector& b) {
    return a * b;
}

Vector PosSymLinSystem::MatrixVectorMultiply(const Matrix& A, const Vector& x) {
    Vector result(mSize);
    for (int i = 1; i <= mSize; i++) {
        double sum = 0.0;
        for (int j = 1; j <= mSize; j++) {
            sum += A(i,j) * x(j);
        }
        result(i) = sum;
    }
    return result;
}