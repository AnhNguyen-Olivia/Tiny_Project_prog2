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
// Transpose the matrix
Matrix Matrix::Transpose() const { 
    Matrix result(mNumCols, mNumRows);              // Swap rows and columns
    for (int i = 1; i <= mNumRows; ++i) {
        for (int j = 1; j <= mNumCols; ++j) {
            result(j, i) = (*this)(i, j);           // Use 1-based indexing
        }
    }
    return result;
}

// Destructor
Matrix::~Matrix() {
    for (int i = 0; i < mNumRows; ++i) {                // Delete each row
        delete[] mData[i];
    }
    delete[] mData;                                     // Delete the array of row pointers
}

// Copy assignment operator
Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) return *this;                   // avoid self-assignment
    for (int i = 0; i < mNumRows; ++i)                  // Clean up
        delete[] mData[i];
    delete[] mData;

    mNumRows = other.mNumRows;                          // copy dimensions
    mNumCols = other.mNumCols;
    mData = new double*[mNumRows];                      //allocate new memory
    for (int i = 0; i < mNumRows; ++i) {
        mData[i] = new double[mNumCols];
        for (int j = 0; j < mNumCols; ++j) {            //copy data element
            mData[i][j] = other.mData[i][j];
        }
    }
    return *this;
}
// Getters
int Matrix::GetNumRows() const { return mNumRows; }
int Matrix::GetNumCols() const { return mNumCols; }

// Access operator ()
double& Matrix::operator()(int i, int j) {
    if (i < 1 || i > mNumRows || j < 1 || j > mNumCols) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
    assert(i >= 1 && i <= mNumRows && j >= 1 && j <= mNumCols);
    return mData[i - 1][j - 1];                         //adjust to 0-based indexing
}

// Const access operator ()
const double& Matrix::operator()(int i, int j) const {
    if (i < 1 || i > mNumRows || j < 1 || j > mNumCols) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
    assert(i >= 1 && i <= mNumRows && j >= 1 && j <= mNumCols);
    return mData[i - 1][j - 1];                         //adjust to 0-based indexing
}
// + operator
Matrix Matrix::operator+(const Matrix& other) const {
    if (mNumRows != other.mNumRows || mNumCols != other.mNumCols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    assert(mNumRows == other.mNumRows && mNumCols == other.mNumCols);
    Matrix result(mNumRows, mNumCols);                  //create result matrix
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[i][j] = mData[i][j] + other.mData[i][j];
    return result;
}

//Subtraction of two matrices (- operator)
Matrix Matrix::operator-(const Matrix& other) const {
    if (mNumRows != other.mNumRows || mNumCols != other.mNumCols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    assert(mNumRows == other.mNumRows && mNumCols == other.mNumCols);
    Matrix result(mNumRows, mNumCols);                      //create result matrix
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[i][j] = mData[i][j] - other.mData[i][j];
    return result;
}

// * scalar
Matrix Matrix::operator*(double scalar) const {
    Matrix result(mNumRows, mNumCols);                      //create result matrix
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[i][j] = mData[i][j] * scalar;
    return result;
}

// * vector
Vector Matrix::operator*(const Vector& vec) const {
    assert(mNumCols == vec.getSize());                      //ensure dimensions match
    Vector result(mNumRows);                                //create result matrix
    for (int i = 0; i < mNumRows; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < mNumCols; ++j)
            result[i] += mData[i][j] * vec[j];              //dot product
    }
    return result;
}

// * matrix
Matrix Matrix::operator*(const Matrix& other) const {
    if (mNumCols != other.mNumRows) {
        throw std::invalid_argument(
            "Matrix multiplication dimension mismatch: " +
            std::to_string(mNumCols) + " != " + 
            std::to_string(other.mNumRows)
        );
    }
    assert(mNumCols == other.mNumRows);
    Matrix result(mNumRows, other.mNumCols);                    //create result matrix
    for (int i = 0; i < mNumRows; ++i) {
        for (int j = 0; j < other.mNumCols; ++j) {
            result.mData[i][j] = 0.0;
            for (int k = 0; k < mNumCols; ++k)
                result.mData[i][j] += mData[i][k] * other.mData[k][j];
        }
    }
    return result;
}

// Unary minus operator
Matrix Matrix::operator-() const {
    Matrix result(mNumRows, mNumCols);                          //create result matrix
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[i][j] = -mData[i][j];
    return result;
}

// Print for debugging
void Matrix::Print() const {
    for (int i = 0; i < mNumRows; ++i) {
        std::cout << "[ ";
        for (int j = 0; j < mNumCols; ++j) {
            std::cout << mData[i][j] << " ";
        }
        std::cout << "]\n";
    }
}

double Matrix::Determinant() const {
    if (!Square()) {
        throw std::domain_error(
            "Determinant only defined for square matrices (current: " +
            std::to_string(mNumRows) + "x" + std::to_string(mNumCols) + ")"
        );
    }
    assert(mNumRows == mNumCols);
    int n = mNumRows;
    if (n == 1) return mData[0][0];                     // Base cases
    if (n == 2) return mData[0][0] * mData[1][1] - mData[0][1] * mData[1][0];
    if (n == 3) {
        return mData[0][0] * (mData[1][1] * mData[2][2] - mData[1][2] * mData[2][1])
                - mData[0][1] * (mData[1][0] * mData[2][2] - mData[1][2] * mData[2][0])
                + mData[0][2] * (mData[1][0] * mData[2][1] - mData[1][1] * mData[2][0]);
    }

    double det = 0.0;                                   //recursive case: Laplace expansion
    for (int p = 0; p < n; ++p) {
        Matrix subMat(n - 1, n - 1);                    //submatrix for minor
        for (int i = 1; i < n; ++i) {
            int colIdx = 0;
            for (int j = 0; j < n; ++j) {
                if (j == p) continue;
                subMat.mData[i - 1][colIdx] = mData[i][j];
                ++colIdx;
            }
        }
        det += ((p % 2 == 0) ? 1 : -1) * mData[0][p] * subMat.Determinant();
    }
    return det;
}

Matrix Matrix::Inverse() const {                        //compute matrix inverse using cofactor method
    assert(mNumRows == mNumCols);                       //must be square
    int n = mNumRows;
    double det = this->Determinant();
    const double EPS = 1e-12;
    if (std::abs(det) < EPS) {
        throw std::runtime_error("Matrix is singular (determinant too small)");
    }
    assert(std::abs(det) > EPS && "Matrix is singular or nearly singular");

    Matrix inv(n, n);                                   //resulting inverse matrix
    if (n == 1) {                                       //base case: 1x1
        inv.mData[0][0] = 1.0 / mData[0][0];
        return inv;
    }
    if (n == 2) {                                       //base case: 2x2
        inv.mData[0][0] =  mData[1][1] / det;
        inv.mData[0][1] = -mData[0][1] / det;
        inv.mData[1][0] = -mData[1][0] / det;
        inv.mData[1][1] =  mData[0][0] / det;
        return inv;
    }
    
    for (int i = 0; i < n; ++i) {                       //for n > 2, use cofactor expansion
        for (int j = 0; j < n; ++j) {
            Matrix minor(n - 1, n - 1);                 //minor matrix
            int rowIdx = 0;
            for (int r = 0; r < n; ++r) {
                if (r == i) continue;
                int colIdx = 0;
                for (int c = 0; c < n; ++c) {
                    if (c == j) continue;
                    minor.mData[rowIdx][colIdx] = mData[r][c];
                    ++colIdx;
                }
                ++rowIdx;
            }
            double cofactor = ((i + j) % 2 == 0 ? 1 : -1) * minor.Determinant();
            inv.mData[j][i] = cofactor / det;           // Transpose for adjugate
        }
    }
    return inv;
}

bool Matrix::Symmetric() const {                        //check if the matrix is symmetric
if (!Square()) return false;                            //not square -> not symmetric
for (int i = 0; i < mNumRows; i++) {
    for (int j = 0; j < i; j++) {
        if (mData[i][j] != mData[j][i]) {
            return false;                               //found asymmetry
        }
    }
}
return true;
}