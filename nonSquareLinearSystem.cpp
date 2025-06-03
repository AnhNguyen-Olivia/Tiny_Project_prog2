#include "nonSquareLinearSystem.h"

NonSquareLinearSystem::NonSquareLinearSystem(const Matrix& A, const Vector& b) 
    : _A(A), _b(b) {
    assert(A.GetNumRows() == b.getSize() && "Dimension mismatch!");
}

Vector NonSquareLinearSystem::SolveWithPseudoInverse() const {
    Matrix A_pseudo = _A.PseudoInverse();
    Matrix b_mat = _b.ToMatrix();
    Matrix result = A_pseudo * b_mat;
    Vector x(result.GetNumRows());
    for (int i = 1; i <= result.GetNumRows(); ++i) {
        x(i) = result(i, 1);
    }
    return x;
}

Vector NonSquareLinearSystem::SolveWithTikhonov(double lambda) const {
    Matrix A_T = _A.Transpose();
    if (_A.GetNumRows() >= _A.GetNumCols()) { // Over-determined
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