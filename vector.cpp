// vector.cpp
#include "vector.h"
#include "matrix.h"
#include <iostream>
#include <cassert>
#include <sstream>
#include <iomanip>

// Constructor
Vector::Vector(int size)
    : mSize(size), mData(size > 0 ? new double[size]() : nullptr) {
    assert(size > 0);
}

// Constructor from array
Vector::Vector(double array[], int size)
    : mSize(size), mData(size > 0 ? new double[size] : nullptr) {
    assert(size > 0);
    for (int i = 0; i < mSize; ++i) {
        mData[i] = array[i];
    }
}

// Copy constructor
Vector::Vector(const Vector& other)
    : mSize(other.mSize), mData(other.mSize > 0 ? new double[other.mSize] : nullptr) {
    for (int i = 0; i < mSize; ++i) {
        mData[i] = other.mData[i];
    }
}

// Destructor
Vector::~Vector() {
    delete[] mData;
    mData = nullptr;
    mSize = 0;
}

// Assignment operator
Vector& Vector::operator=(const Vector& other) {
    if (this == &other) return *this;
    delete[] mData;
    mSize = other.mSize;
    mData = mSize > 0 ? new double[mSize] : nullptr;
    for (int i = 0; i < mSize; ++i) {
        mData[i] = other.mData[i];
    }
    return *this;
}

// Accessors
int Vector::getSize() const { return mSize; }
double* Vector::getData() const { return mData; }

// [] operator (0-based)
double& Vector::operator[](int index) {
    assert(index >= 0 && index < mSize);
    return mData[index];
}

// [] operator (0-based) const
const double& Vector::operator[](int index) const {
    assert(index >= 0 && index < mSize);
    return mData[index];
}

// () operator (1-based)
double& Vector::operator()(int index) {
    assert(index >= 1 && index <= mSize);
    return mData[index - 1];
}

// () operator (1-based) const
const double& Vector::operator()(int index) const {
    assert(index >= 1 && index <= mSize);
    return mData[index - 1];
}

// Unary minus
Vector Vector::operator-() const {
    Vector result(mSize);
    for (int i = 0; i < mSize; ++i)
        result.mData[i] = -mData[i];
    return result;
}

// Prefix increment
Vector& Vector::operator++() {
    for (int i = 0; i < mSize; ++i)
        ++mData[i];
    return *this;
}

// Postfix increment
Vector Vector::operator++(int) {
    Vector temp(*this);
    ++(*this);
    return temp;
}

// Prefix decrement
Vector& Vector::operator--() {
    for (int i = 0; i < mSize; ++i)
        --mData[i];
    return *this;
}

// Postfix decrement
Vector Vector::operator--(int) {
    Vector temp(*this);
    --(*this);
    return temp;
}

// + operator
Vector Vector::operator+(const Vector& other) const {
    assert(mSize == other.mSize);
    Vector result(mSize);
    for (int i = 0; i < mSize; ++i)
        result.mData[i] = mData[i] + other.mData[i];
    return result;
}

// += operator
Vector& Vector::operator+=(const Vector& other) {
    assert(mSize == other.mSize);
    for (int i = 0; i < mSize; ++i)
        mData[i] += other.mData[i];
    return *this;
}

// - operator
Vector Vector::operator-(const Vector& other) const {
    assert(mSize == other.mSize);
    Vector result(mSize);
    for (int i = 0; i < mSize; ++i)
        result.mData[i] = mData[i] - other.mData[i];
    return result;
}

// -= operator
Vector& Vector::operator-=(const Vector& other) {
    assert(mSize == other.mSize);
    for (int i = 0; i < mSize; ++i)
        mData[i] -= other.mData[i];
    return *this;
}

// * scalar multiplication
Vector Vector::operator*(double scalar) const {
    Vector result(mSize);
    for (int i = 0; i < mSize; ++i)
        result.mData[i] = mData[i] * scalar;
    return result;
}
Vector& Vector::operator*=(double scalar) {
    for (int i = 0; i < mSize; ++i)
        mData[i] *= scalar;
    return *this;
}

// Dot product
double Vector::operator*(const Vector& other) const {
    assert(mSize == other.mSize);
    double sum = 0.0;
    for (int i = 0; i < mSize; ++i)
        sum += mData[i] * other.mData[i];
    return sum;
}

// Utility: toString
std::string Vector::toString() const {
    std::ostringstream oss;
    oss << "[ ";
    for (int i = 0; i < mSize; ++i) {
        oss << std::fixed << std::setprecision(2) << mData[i];
        if (i < mSize - 1) oss << ", ";
    }
    oss << " ]";
    return oss.str();
}

// Print vector for debugging
void Vector::Print() const {
    std::cout << toString() << std::endl;
}

// Conversion to Matrix
Matrix Vector::ToMatrix() const {
    Matrix mat(mSize, 1);
    for (int i = 1; i <= mSize; ++i)
        mat(i, 1) = (*this)(i);
    return mat;
}


