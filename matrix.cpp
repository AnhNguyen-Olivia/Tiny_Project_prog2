#include <iostream>
#include <initializer_list>
#include <cassert>
using namespace std;

template <unsigned Rows, unsigned Cols, typename T = float>
class Matrix {
public:
    T data[Rows * Cols];
    Matrix() = default;
    Matrix(initializer_list<T> values) {
        assert(values.size() == Rows * Cols && "Initializer list size mismatch");
        copy(values.begin(), values.end(), data);
    }
    Matrix(initializer_list<initializer_list<T>> values) {
        assert(values.size() == Rows && "Row count mismatch");
        unsigned idx = 0;
        for (const auto& row : values) {
            assert(row.size() == Cols && "Column count mismatch in row");
            for (const auto& val : row) {
                data[idx++] = val;
            }
        }
    }

    T& operator()(unsigned row, unsigned col) { return data[row * Cols + col]; }
    const T& operator()(unsigned row, unsigned col) const { return data[row * Cols + col]; }

    void print() const {
        for (unsigned i = 0; i < Rows; ++i) {
            for (unsigned j = 0; j < Cols; ++j) {
                cout << (*this)(i, j) << " ";
            }
            cout << "\n";
        }
    }
};
template <unsigned R1, unsigned C1R2, unsigned C2, typename T>
Matrix<R1, C2, T> operator*(const Matrix<R1, C1R2, T>& lhs, const Matrix<C1R2, C2, T>& rhs) {
    Matrix<R1, C2, T> result;
    for (unsigned i = 0; i < R1; ++i) {
        for (unsigned j = 0; j < C2; ++j) {
            T sum = 0;
            for (unsigned k = 0; k < C1R2; ++k) {
                sum += lhs(i, k) * rhs(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

int main() {
    Matrix<2, 3> a = {{1, 2, 3},
                      {4, 5, 6}};

    Matrix<3, 2> b = {{1, 2},
                      {3, 4},
                      {5, 6}};

    cout << "Matrix A:\n";
    a.print();
    cout << "\nMatrix B:\n";
    b.print();

    auto c = a * b;
    cout << "\nResult (A * B):\n";
    c.print();

    return 0;
}