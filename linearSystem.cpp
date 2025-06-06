  // add the necessary includes
#include "linearSystem.h"
#include "vector.h"
#include "matrix.h"
#include <cassert>         // to use assert for debugging
#include <cmath>           // fix absolute value calculation
#include <algorithm>       // for std::swap

// ------------------- LinearSystem implementation -------------------
LinearSystem::LinearSystem(const Matrix& A, const Vector& b) {
    if (A.GetNumRows() != A.GetNumCols()) {                                 // check if matrix A is square
        throw std::invalid_argument("Matrix must be square");
    }
    
    if (A.GetNumRows() != b.getSize()) {                                    // check if vector b has the same size as matrix A
        throw std::invalid_argument("Matrix and vector size mismatch");
    }
    mSize = A.GetNumRows();                                                 // initialize size and allocate memory for matrix and vector
    
    // dynamically allocate memory for matrix and vector
    mpA = new Matrix(A);
    mpb = new Vector(b);
}

// Destructor: free allocated memory
LinearSystem::~LinearSystem() {
    delete mpA;    // Delete the dynamically allocated matrix
    delete mpb;    // Delete the dynamically allocated vector
}

// using Gaussian elimination with partial pivoting to solve Ax = b
Vector LinearSystem::Solve() {
    int n = mSize;                 // Get the size of the system
    Matrix A(*mpA);                // Create a copy of the matrix A
    Vector b(*mpb);                // Create a copy of the vector b

    // Forward elimination with partial pivoting
    for (int k = 0; k < n - 1; ++k) {
    // Find pivot - the row with the largest absolute value in column k
        int pivot = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(A(i + 1, k + 1)) > std::abs(A(pivot + 1, k + 1))) { // Compare absolute values for pivoting
                pivot = i;
            }
        }
        
        // Swap rows if a better pivot is found
        if (pivot != k) {
            for (int j = 1; j <= n; ++j) std::swap(A(k + 1, j), A(pivot + 1, j));
            std::swap(b(k + 1), b(pivot + 1));
        }

        // Elimination step: zero out elements below diagonal in column k
        for (int i = k + 1; i < n; ++i) {
            double factor = A(i + 1, k + 1) / A(k + 1, k + 1);  // Calculate elimination factor
            for (int j = k + 1; j <= n; ++j) {
                A(i + 1, j) -= factor * A(k + 1, j);           // Update matrix elements
            }
            b(i + 1) -= factor * b(k + 1);                     // Update right-hand side vector
        }
    }

    // Back substitution to find solution vector x
    Vector x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = b(i + 1);                                // Start with RHS value
        for (int j = i + 1; j < n; ++j) {
            sum -= A(i + 1, j + 1) * x(j + 1);               // Subtract known terms
        }
        x(i + 1) = sum / A(i + 1, i + 1);                    // Solve for unknown
    }

    return x;  // Return solution vector
}

// ------------------- PosSymLinSystem implementation -------------------
// Constructor for Positive Symmetric Linear System (specialized subclass)
PosSymLinSystem::PosSymLinSystem(Matrix& A, Vector& b) : LinearSystem(A, b) {
    // Verify that matrix is symmetric (required for conjugate gradient method)
    if (!A.Symmetric()) {
        throw std::invalid_argument("Matrix must be symmetric for PosSymLinSystem");
    }
}

// Solve using Conjugate Gradient method (more efficient for positive definite symmetric matrices)
Vector PosSymLinSystem::Solve() {
    // Get references to the matrix and vector
    const Matrix& A = *mpA;
    const Vector& b = *mpb;
    
    // Algorithm parameters
    const int maxIterations = 1000;       // Maximum number of iterations
    const double tolerance = 1e-10;       // Convergence criterion
    
    Vector x(mSize);                      // Initial guess (all zeros)
    Vector r = b - MatrixVectorMultiply(A, x);  // Initial residual r = b - Ax
    Vector p = r;                         // Initial search direction
    double rsold = DotProduct(r, r);      // r dot r (squared norm of residual)
    
    // Main conjugate gradient iteration loop
    for (int i = 0; i < maxIterations; i++) {
        Vector Ap = MatrixVectorMultiply(A, p);     // Calculate A*p
        double alpha = rsold / DotProduct(p, Ap);   // Step length
        x = x + p * alpha;                          // Update solution
        r = r - Ap * alpha;                         // Update residual
        
        double rsnew = DotProduct(r, r);            // Calculate new squared residual norm
        if (sqrt(rsnew) < tolerance) {              // Check convergence
            break;                                  // Exit if converged
        }
        
        p = r + p * (rsnew / rsold);                // Update search direction
        rsold = rsnew;                              // Store current residual for next iteration
    }
    
    return x;  // Return solution vector
}

// Helper method: Calculate dot product between two vectors
double PosSymLinSystem::DotProduct(const Vector& a, const Vector& b) {
    return a * b;  // Use Vector's overloaded * operator for dot product
}

// Helper method: Matrix-vector multiplication
Vector PosSymLinSystem::MatrixVectorMultiply(const Matrix& A, const Vector& x) {
    Vector result(mSize);                      // Initialize result vector
    for (int i = 1; i <= mSize; i++) {         // For each row
        double sum = 0.0;                      // Initialize row sum
        for (int j = 1; j <= mSize; j++) {     // For each column
            sum += A(i,j) * x(j);              // Multiply and accumulate
        }
        result(i) = sum;                       // Store result in output vector
    }
    return result;
}