#include "nonSquareLinearSystem.h"

NonSquareLinearSystem::NonSquareLinearSystem(const Matrix& A, const Vector& b) 
    : _A(A), _b(b) {
    if (A.GetNumRows() != b.getSize()) {
        throw std::invalid_argument("Dimension mismatch!");
    };
}

Vector NonSquareLinearSystem::SolveWithPseudoInverse() const {
    if (_A.GetNumRows() > _A.GetNumCols()) {
        // Over-determined: x = (A^T A)^-1 A^T b
        Matrix A_T = _A.Transpose();
        Matrix ATA = A_T * _A;
        Vector ATb = A_T * _b;
        LinearSystem ls(ATA, ATb);
        return ls.Solve();
    } else {
        // Under-determined: x = A^T (A A^T)^-1 b
        Matrix A_T = _A.Transpose();
        Matrix AAT = _A * A_T;
        LinearSystem ls(AAT, _b);
        Vector y = ls.Solve();
        return A_T * y;
    }
}

Vector NonSquareLinearSystem::SolveWithTikhonov(double lambda) const {
    Matrix A_T = _A.Transpose();
    if (_A.GetNumRows() > _A.GetNumCols()) { // Over-determined
        Matrix regularized = A_T * _A + Matrix::IdentityMatrix(_A.GetNumCols()) * lambda;
        Vector rhs = A_T * _b;
        LinearSystem ls(regularized, rhs);
        return ls.Solve();
    } else { // Under-determined
        Matrix regularized = _A * A_T + Matrix::IdentityMatrix(_A.GetNumRows()) * lambda;
        LinearSystem ls(regularized, _b);
        Vector y = ls.Solve();
        return A_T * y;
    }
}