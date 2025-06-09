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
