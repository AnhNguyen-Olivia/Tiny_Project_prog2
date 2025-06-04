#ifndef MATRIX_H
#define MATRIX_H

#include <cassert>
#include <iostream>

class Vector;

// Matrix class for handling 2D arrays of doubles
class Matrix {      
private:
    int mNumRows;       // Number of rows in the matrix
    int mNumCols;       // Number of columns in the matrix
    double** mData;     // Pointer to the data array (2D array)

// Constructors & Destructor
public: 
    Matrix(int numRows, int numCols); // Constructor with specified number of rows and columns
    Matrix(const Matrix& other);      // Copy constructor
    ~Matrix();                        // Destructor

    Matrix& operator=(const Matrix& other); // Copy assignment operator

    // Accessors
    int GetNumRows() const;         // Get number of rows
    int GetNumCols() const;         // Get number of columns


    // Access operator for 1-based indexing
    double& operator()(int i, int j);               // Access operator for non-const access
    const double& operator()(int i, int j) const;   // Access operator for const access

    // Unary operators
    Matrix operator+(const Matrix& other) const;    // Addition
    Matrix operator-(const Matrix& other) const;    // Subtraction
    Matrix operator*(double scalar) const;          // Scalar multiplication
    Vector operator*(const Vector& vec) const;      // Vector multiplication
    Matrix operator*(const Matrix& other) const;    // Matrix multiplication

    // Utility functions
    Matrix Transpose() const;                   // Transpose the matrix
    Matrix PseudoInverse() const;               // Pseudo-inverse of the matrix
    static Matrix IdentityMatrix(int size);     // Create an identity matrix of given size
    Matrix operator-() const;                   // Negation operator
    double Determinant() const;                 // Calculate the determinant of the matrix
    Matrix Inverse() const;                     // Calculate the inverse of the matrix
    bool Square() const { return mNumRows == mNumCols; }  // Check the square property of the matrix
    bool Symmetric() const;                     // Check the symmetric property of the matrix

    void Print() const;                         // Print the matrix for debugging
};

#endif