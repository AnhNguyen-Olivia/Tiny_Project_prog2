// vector.h
#ifndef VECTOR_H
#define VECTOR_H

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

// Forward declaration if matrix is used in ToMatrix()
class matrix;

class Vector {
private:            
    int mSize;         // Size of the vector
    double* mData;     // Pointer to the data array

public:
    // Constructors & Destructor
    Vector(int size = 0);                        // Default/size constructor
    Vector(double array[], int size);            // From array
    Vector(const Vector& other);                 // Copy constructor
    ~Vector();                                   // Destructor

    // Assignment
    Vector& operator=(const Vector& other);      // Copy assignment  

    // Accessors
    int getSize() const;                         // Get size of the vector
    double* getData() const;                     // Get raw data pointer

    // Element access
    double& operator[](int index);               // Subscript operator for non-const access
    const double& operator[](int index) const;   // Subscript operator for const access
    double& operator()(int index);               // Parentheses operator for non-const access
    const double& operator()(int index) const;   // Parentheses operator for const access

    // Unary operators
    Vector operator-() const;        // Negation
    Vector& operator++();            // Prefix increment
    Vector operator++(int);          // Postfix increment
    Vector& operator--();            // Prefix decrement
    Vector operator--(int);          // Postfix decrement

    // Binary operators
    Vector operator+(const Vector& other) const;       // Addition
    Vector& operator+=(const Vector& other);           // Addition assignment
    Vector operator-(const Vector& other) const;       // Subtraction
    Vector& operator-=(const Vector& other);           // Subtraction assignment
    Vector operator*(double scalar) const;             // Scalar multiplication
    Vector& operator*=(double scalar);                 // Scalar multiplication assignment

    // Dot product
    double operator*(const Vector& other) const;       // Dot product

    // Utility
    std::string toString() const;                      // Convert to string representation
    void Print() const;                                // Print to standard output   

    // Conversion
    Matrix ToMatrix() const;                           // Convert to Matrix object
};

#endif