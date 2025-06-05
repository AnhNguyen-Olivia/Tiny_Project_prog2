#include "vector.h"
#include "matrix.h"
#include <iostream>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <stdexcept> // For std::invalid_argument

// Constructor: initialize a vector with given size, all elements set to 0
Vector::Vector(int size)
    : mSize(size), mData(size > 0 ? new double[size]() : nullptr) {
    if (size <= 0) {
        throw std::invalid_argument("Vector size must be positive");
    }
}

// Constructor from array: copy data from input array to new vector
Vector::Vector(double array[], int size)
    : mSize(size), mData(size > 0 ? new double[size] : nullptr) {
    if (size <= 0) {
        throw std::invalid_argument("Vector size must be positive");
    }
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

// Destructor: release allocated memory
Vector::~Vector() {
    delete[] mData;
    mData = nullptr;
    mSize = 0;
}

// Assignment operator
Vector& Vector::operator=(const Vector& other) {
    if (this == &other) return *this; // Self-assignment check
    delete[] mData;
    mSize = other.mSize;
    mData = mSize > 0 ? new double[mSize] : nullptr;
    for (int i = 0; i < mSize; ++i) {
        mData[i] = other.mData[i];
    }
    return *this;
}

// Accessors: 
int Vector::getSize() const { return mSize; }               //Get the number of elements in the vector
double* Vector::getData() const { return mData; }           // Get pointer to raw data (use with caution)


// [] operator (0-based)
double& Vector::operator[](int index) {
    if (index < 0 || index >= mSize) {                      //bonus checking 
        throw std::out_of_range("Vector index out of range");
    }
    return mData[index];
}

// [] operator (0-based) const
const double& Vector::operator[](int index) const {
    if (index < 0 || index >= mSize) {
        throw std::out_of_range("Vector index out of range");
    }
    return mData[index];
}

// () operator (1-based)
double& Vector::operator()(int index) {
    if (index < 1 || index > mSize) {
        throw std::out_of_range("Vector 1-based index out of range");
    }
    return mData[index - 1];                                //convert to based-0 -> return const
}

// () operator (1-based) const
const double& Vector::operator()(int index) const {
    if (index < 1 || index > mSize) {
        throw std::out_of_range("Vector 1-based index out of range");
    }
    return mData[index - 1];
}

// Unary minus --> negates all elements
Vector Vector::operator-() const {
    Vector result(mSize);                                  //create the new vector at the same size
    for (int i = 0; i < mSize; ++i)                        //iterate all element  
        result.mData[i] = -mData[i];                       //implement the unary minus operate
    return result;
}

// Prefix increment--> (++i)
Vector& Vector::operator++() {
    for (int i = 0; i < mSize; ++i)                       //increase all element 
        ++mData[i];                                       
    return *this;                                         // return reference to modify "this"
}

// Postfix increment
Vector Vector::operator++(int) {                          // "int" to distinguishes from prefix
    Vector temp(*this);                                   // save the current state
    ++(*this);                                            // use prefix increase 
    return temp;                                          // return old state
}

// Prefix decrement
Vector& Vector::operator--() {                            //similar to the prefix and postfix increasement
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

// Vector addition operator (+ operator)
Vector Vector::operator+(const Vector& other) const {
    if (mSize != other.mSize) {                             //check the size of two vector                                 
        throw std::invalid_argument("Vector sizes do not match for addition");
    }                                                       // if the size is not same-->throw them
    Vector result(mSize);                                   //create the new vector for result
    for (int i = 0; i < mSize; ++i)                         //iterate each element
        result.mData[i] = mData[i] + other.mData[i];        //add corresponding vector from both vectors
    return result;                                          
}

// Vector addition assignment operator(+= operator)
Vector& Vector::operator+=(const Vector& other) {           
    if (mSize != other.mSize) {                             //check the size of two vector 
        throw std::invalid_argument("Vector sizes do not match for addition");
    }                                                       // if the size is not same-->throw them
    for (int i = 0; i < mSize; ++i)                         //iterate each element
        mData[i] += other.mData[i];                         //add other vector to current vector
    return *this;
}

//Vector subtraction operator(- operator)
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


