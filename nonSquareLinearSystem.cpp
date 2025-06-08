#include "nonSquareLinearSystem.h"  // Include the header file for the NonSquareLinearSystem class

// Constructor for NonSquareLinearSystem
NonSquareLinearSystem::NonSquareLinearSystem(const Matrix& A, const Vector& b) 
    : _A(A), _b(b)                                          // Initialize the private members _A and _b with given matrix A and vector b
{
    
    if (A.GetNumRows() != b.getSize()) {                     // Check if the number of rows in A matches the size of vector b
        throw std::invalid_argument("Dimension mismatch!");  // Throw an exception if dimensions are incompatible
    };
}


Vector NonSquareLinearSystem::SolveWithPseudoInverse() const {  // Solve the non-square system using Moore-Penrose pseudo-inverse
    if (_A.GetNumRows() > _A.GetNumCols()) {                    // Case: Over-determined system (more equations than unknowns)
        Matrix A_T = _A.Transpose();                            // Compute the transpose of A
        Matrix ATA = A_T * _A;                                  // Compute A^T * A
        Vector ATb = A_T * _b;                                  // Compute A^T * b
        LinearSystem ls(ATA, ATb);                              // Create a square linear system with ATA and ATb
        return ls.Solve();                                      // Solve the linear system and return the result
    } else {                                                    // Case: Under-determined system (more unknowns than equations)
        Matrix A_T = _A.Transpose();                            // Compute the transpose of A
        Matrix AAT = _A * A_T;                                  // Compute A * A^T
        LinearSystem ls(AAT, _b);                               // Create a square linear system with AAT and b
        Vector y = ls.Solve();                                  // Solve the linear system to get intermediate vector y
        return A_T * y;                                         // Return A^T * y as the solution
    }
}

// Solve the system using Tikhonov regularization (also known as Ridge Regression)
Vector NonSquareLinearSystem::SolveWithTikhonov(double lambda) const {
    Matrix A_T = _A.Transpose();                                // Compute the transpose of A
    if (_A.GetNumRows() > _A.GetNumCols()) {                    // Case: Over-determined system
        Matrix regularized = A_T * _A + Matrix::IdentityMatrix(_A.GetNumCols()) * lambda;       // Compute the regularized matrix: A^T A + λI
        Vector rhs = A_T * _b;                                  // Compute the right-hand side: A^T * b
        LinearSystem ls(regularized, rhs);                      // Create and solve the regularized system
        return ls.Solve();
    } else {                                                    // Case: Under-determined system
        Matrix regularized = _A * A_T + Matrix::IdentityMatrix(_A.GetNumRows()) * lambda;       // Compute the regularized matrix: A A^T + λI
        LinearSystem ls(regularized, _b);                       // Create and solve the regularized system
        Vector y = ls.Solve();                                  // Solve the system to get intermediate vector y
        return A_T * y;                                         // Return A^T * y as the solution
    }
}
