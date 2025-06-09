#include "vector.h"
#include "matrix.h"
#include "linearSystem.h"

// Constructor
Matrix::Matrix(int numRows, int numCols) {
    if (numRows <= 0 || numCols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    assert(numRows > 0 && numCols > 0); // Ensure dimensions are positive
    mNumRows = numRows;
    mNumCols = numCols;

    mData = new double*[mNumRows];
    for (int i = 0; i < mNumRows; ++i) {
        mData[i] = new double[mNumCols];
        for (int j = 0; j < mNumCols; ++j) {
            mData[i][j] = 0.0;
        }
    }
}
// Copy constructor
Matrix::Matrix(const Matrix& other) {
    // numRows and numCols are PARAMETERS (local to this function)
    // mNumRows and mNumCols are MEMBER VARIABLES (part of the object)

    mNumRows = other.mNumRows;                      //store parameter value in member variable
    mNumCols = other.mNumCols;                      //store parameter value in member variable

    mData = new double*[mNumRows];                  //allocate memory to for an array of row pointer
    for (int i = 0; i < mNumRows; ++i) {            //loop row and allocate columns
        mData[i] = new double[mNumCols];
        for (int j = 0; j < mNumCols; ++j) {        //initialize element in row to 0
            mData[i][j] = other.mData[i][j];        //copy data from other to current data
        }
    }
}

// Static method to create an identity matrix
Matrix Matrix::IdentityMatrix(int size) {
    Matrix I(size, size);                           //create the square matrix 
    for (int i = 1; i <= size; ++i) {               // loop through the diagonal matrix
        I(i, i) = 1.0;                              // Diagonal entries set to 1.0
    }
    return I;
}

// Compute the pseudo-inverse using the Moore-Penrose method
Matrix Matrix::PseudoInverse() const { 
    if (mNumRows >= mNumCols) {                     // over-determined: (A^T A)^-1 A^T
        Matrix A_T = this->Transpose();             // compute the tranpose of A
        Matrix ATA = A_T * (*this);                 // compute A^T * A
        Matrix ATA_inv(mNumCols, mNumCols);         //create square matrix to store the inverse ATA

        for (int i = 1; i <= mNumCols; ++i) {       //solve (A^T A)* x = e for each columns
            Vector e(mNumCols);                     
            e(i) = 1.0;                             // set unit vector for solving
            LinearSystem ls(ATA, e);                
            Vector col = ls.Solve();
            for (int j = 1; j <= mNumCols; ++j) {   //store solution in the inverse matrix
                ATA_inv(j, i) = col(j);
            }
        }
        return ATA_inv * A_T;                       // compute the final (A^T A)^-1 A^T
    } 
    
    else {                                          // Under-determined: A^T (A A^T)^-1
        Matrix A_T = this->Transpose();             // compute transpose of A
        Matrix AAT = (*this) * A_T;                 // compute A* A^T
        Matrix AAT_inv(mNumRows, mNumRows);         //create square matrix to store inverse AAT
        for (int i = 1; i <= mNumRows; ++i) {       //solve (A A^T)* x = e for each columns
            Vector e(mNumRows);
            e(i) = 1.0;                             //set unit = 1
            LinearSystem ls(AAT, e);
            Vector col = ls.Solve();
            for (int j = 1; j <= mNumRows; ++j) {   //store solution in inverse matrix
                AAT_inv(j, i) = col(j);
            }
        }
        return (A_T * AAT_inv).Transpose();         // Compute the final A^T (A A^T)^-1--> transpose the result
    }
}
    
