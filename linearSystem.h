#ifndef LINEARSYSTEM_H
#define LINEARSYSTEM_H

#include "matrix.h"
#include "vector.h"

// Class for solving linear systems of equations Ax = b
class LinearSystem { 
    
protected:
    int mSize;
    Matrix* mpA;
    Vector* mpb;

// Constructors & Destructor
public: 
    LinearSystem(const Matrix& A, const Vector& b); // Constructor with matrix A and vector b
    virtual ~LinearSystem();                        // Destructor

    // Solve Ax = b using Gaussian elimination
    virtual Vector Solve();

private:
    LinearSystem();                          // prevent default constructor
    LinearSystem(const LinearSystem& other); // prevent copy constructor
};

class PosSymLinSystem : public LinearSystem {
public:
    // Constructor
    PosSymLinSystem(Matrix& A, Vector& b);
    
    // Override Solve method to use Conjugate Gradient
    Vector Solve() override;

private:
    // Helper methods for Conjugate Gradient
    double DotProduct(const Vector& a, const Vector& b);
    Vector MatrixVectorMultiply(const Matrix& A, const Vector& x);
};
#endif