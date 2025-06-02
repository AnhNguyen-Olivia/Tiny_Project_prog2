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

#endif