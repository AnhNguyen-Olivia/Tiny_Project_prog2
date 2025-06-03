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
