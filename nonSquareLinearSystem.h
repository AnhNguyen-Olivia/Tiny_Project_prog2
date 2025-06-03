#include "matrix.h"
#include "vector.h"
#include "linearSystem.h"

class NonSquareLinearSystem {
private:
    Matrix _A;
    Vector _b;

public:
    NonSquareLinearSystem(const Matrix& A, const Vector& b);
    Vector SolveWithPseudoInverse() const;
    Vector SolveWithTikhonov(double lambda) const;
};