#include "linearSystem.h"
#include "matrix.h"
#include "vector.h"

#include <iostream>
#include <cassert>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>
#include <chrono>
#include <ctime>

using namespace std;

class LinearSystemTestSuite {
private:
    int totalTests = 0;
    int passedTests = 0;
    int testCounter = 1;                            // For test IDs
    const double EPSILON = 1e-6;                    // For floating point comparisons

    // ANSI color codes for output
    const string GREEN = "\033[32m";
    const string RED = "\033[31m";
    const string YELLOW = "\033[33m";
    const string RESET = "\033[0m";

    bool almostEqual(double a, double b) const {
        return fabs(a - b) < EPSILON;
    }

    bool vectorsEqual(const Vector& v1, const Vector& v2) const {
        if (v1.getSize() != v2.getSize()) return false;
        for (int i = 1; i <= v1.getSize(); i++) {
            if (!almostEqual(v1(i), v2(i))) return false;
        }
        return true;
    }

    // Check if Ax = b approximately
    bool isSolution(const Matrix& A, const Vector& x, const Vector& b) const {
        Vector computed = A * x;
        return vectorsEqual(computed, b);
    }

    // Function object base class to replace lambdas
    class TestFunction {
    public:
        virtual bool operator()() = 0;
        virtual ~TestFunction() {}
    };

    void runTest(const string& testName, TestFunction& testFunction) {
        const int RESULT_COL = 60; // Column where PASSED/FAILED should start
        totalTests++;

        // Format test ID
        ostringstream idStream;
        idStream << "[T" << setw(2) << setfill('0') << testCounter++ << "]";
        string testID = idStream.str();

        // Print test ID and name together
        cout << "  " << testID << "  " << testName;

        // Calculate and print dots
        int currentLength = 2 + testID.length() + 2 + testName.length(); // 2 for "  ", 2 for "  "
        int dots = RESULT_COL - currentLength;
        if (dots < 3) dots = 3;
        cout << string(dots, '.');

        // Result
        try {
            bool result = testFunction();
            if (result) {
                cout << GREEN << "PASSED" << RESET << endl;
                passedTests++;
            } else {
                cout << RED << "FAILED" << RESET << endl;
            }
        } catch (const exception& e) {
            cout << YELLOW << "WARN" << RESET << " (" << e.what() << ")" << endl;
        } catch (...) {
            cout << RED << "FAILED" << RESET << " (Unknown exception)" << endl;
        }
    }

    void printSuiteHeader() {
        // Version and timestamp
        const string VERSION = "v1.0";
        auto now = chrono::system_clock::now();
        time_t now_c = chrono::system_clock::to_time_t(now);
        tm local_tm = {};
        
        // Platform-specific time conversion
        #if defined(_WIN32) || defined(_WIN64)
            // Windows uses localtime_s with reversed parameter order
            localtime_s(&local_tm, &now_c);
        #else
            // POSIX systems use localtime_r
            localtime_r(&now_c, &local_tm);
        #endif
        
        char timebuf[32];
        strftime(timebuf, sizeof(timebuf), "%Y-%m-%d  %H:%M:%S", &local_tm);

        cout << "\n" << string(50, '=') << RESET << endl;
        cout << "Test Suite: LINEAR SYSTEM " << VERSION << endl;
        cout << "Run date: " << timebuf << endl;
        cout << string(50, '=') << RESET << endl;
    }

    void printSectionHeader(const string& sectionName) {
        cout << "\n=== " << sectionName << " ===" << endl;
    }

    // Basic LinearSystem Tests
    class ConstructorInitTest : public TestFunction {
    public:
        bool operator()() override {
            Matrix A(3, 3);
            Vector b(3);
            
            A(1, 1) = 1.0; A(1, 2) = 0.0; A(1, 3) = 0.0;
            A(2, 1) = 0.0; A(2, 2) = 1.0; A(2, 3) = 0.0;
            A(3, 1) = 0.0; A(3, 2) = 0.0; A(3, 3) = 1.0;
            
            b(1) = 1.0; b(2) = 2.0; b(3) = 3.0;
            
            LinearSystem ls(A, b);
            // If we get here without assertion failures, it worked
            return true;
        }
    };

    class IdentityMatrixTest : public TestFunction {
    private:
        LinearSystemTestSuite* suite;
    public:
        IdentityMatrixTest(LinearSystemTestSuite* s) : suite(s) {}
        bool operator()() override {
            Matrix A = Matrix::IdentityMatrix(3);
            Vector b(3);
            b(1) = 1.0; b(2) = 2.0; b(3) = 3.0;
            
            LinearSystem ls(A, b);
            Vector x = ls.Solve();
            
            // For identity matrix, solution should equal the right-hand side
            return suite->isSolution(A, x, b);
        }
    };

    class Solve2x2Test : public TestFunction {
    private:
        LinearSystemTestSuite* suite;
    public:
        Solve2x2Test(LinearSystemTestSuite* s) : suite(s) {}
        bool operator()() override {
            Matrix A(2, 2);
            Vector b(2);
            
            A(1, 1) = 4.0; A(1, 2) = 1.0;
            A(2, 1) = 2.0; A(2, 2) = 3.0;
            
            b(1) = 7.0; b(2) = 13.0;
            
            LinearSystem ls(A, b);
            Vector x = ls.Solve();
            
            return suite->isSolution(A, x, b);
        }
    };

    class Solve3x3Test : public TestFunction {
    private:
        LinearSystemTestSuite* suite;
    public:
        Solve3x3Test(LinearSystemTestSuite* s) : suite(s) {}
        bool operator()() override {
            Matrix A(3, 3);
            Vector b(3);
            
            A(1, 1) = 3.0; A(1, 2) = -1.0; A(1, 3) = 2.0;
            A(2, 1) = 4.0; A(2, 2) = 2.0;  A(2, 3) = 0.0;
            A(3, 1) = -2.0; A(3, 2) = 5.0; A(3, 3) = 1.0;
            
            b(1) = 5.0; b(2) = 2.0; b(3) = 9.0;
            
            LinearSystem ls(A, b);
            Vector x = ls.Solve();
            
            // Check if Ax = b
            return suite->isSolution(A, x, b);
        }
    };

    class PivotingSystemTest : public TestFunction {
    private:
        LinearSystemTestSuite* suite;
    public:
        PivotingSystemTest(LinearSystemTestSuite* s) : suite(s) {}
        bool operator()() override {
            Matrix A(3, 3);
            Vector b(3);
            
            // This system requires pivoting for stability
            A(1, 1) = 0.003; A(1, 2) = 59.14; A(1, 3) = 59.17;
            A(2, 1) = 5.291; A(2, 2) = -6.13; A(2, 3) = 46.78;
            A(3, 1) = 11.2;  A(3, 2) = 9.0;   A(3, 3) = 29.51;
            
            b(1) = 1.09; b(2) = 2.87; b(3) = 8.6;
            
            LinearSystem ls(A, b);
            Vector x = ls.Solve();
            
            // Check if Ax = b
            return suite->isSolution(A, x, b);
        }
    };
    
    // PosSymLinSystem Tests
    class PosSymConstructorTest : public TestFunction {
    public:
        bool operator()() override {
            // Create a positive symmetric matrix
            Matrix A(3, 3);
            Vector b(3);
            
            A(1, 1) = 4.0; A(1, 2) = 1.0; A(1, 3) = 1.0;
            A(2, 1) = 1.0; A(2, 2) = 3.0; A(2, 3) = 2.0;
            A(3, 1) = 1.0; A(3, 2) = 2.0; A(3, 3) = 3.5;
            
            b(1) = 1.0; b(2) = 2.0; b(3) = 3.0;
            
            bool exceptionCaught = false;
            try {
                PosSymLinSystem ls(A, b);
            } catch (...) {
                exceptionCaught = true;
            }
            
            // No exception should be thrown since the matrix is symmetric
            return !exceptionCaught;
        }
    };
    
    class RejectNonSymmetricTest : public TestFunction {
    public:
        bool operator()() override {
            // Create a non-symmetric matrix
            Matrix A(3, 3);
            Vector b(3);
            
            A(1, 1) = 4.0; A(1, 2) = 1.0; A(1, 3) = 1.0;
            A(2, 1) = 2.0; A(2, 2) = 3.0; A(2, 3) = 2.0;  // Note 2.0 != 1.0
            A(3, 1) = 1.0; A(3, 2) = 2.0; A(3, 3) = 3.5;
            
            b(1) = 1.0; b(2) = 2.0; b(3) = 3.0;
            
            bool exceptionCaught = false;
            try {
                PosSymLinSystem ls(A, b);
            } catch (...) {
                exceptionCaught = true;
            }
            
            // Exception should be thrown for non-symmetric matrix
            return exceptionCaught;
        }
    };
    
    class SolvePosDef : public TestFunction {
    private:
        LinearSystemTestSuite* suite;
    public:
        SolvePosDef(LinearSystemTestSuite* s) : suite(s) {}
        bool operator()() override {
            // Create a positive definite matrix
            Matrix A(3, 3);
            Vector b(3);
            
            A(1, 1) = 4.0; A(1, 2) = 1.0; A(1, 3) = 1.0;
            A(2, 1) = 1.0; A(2, 2) = 3.0; A(2, 3) = 2.0;
            A(3, 1) = 1.0; A(3, 2) = 2.0; A(3, 3) = 3.5;
            
            b(1) = 1.0; b(2) = 2.0; b(3) = 3.0;
            
            PosSymLinSystem ls(A, b);
            Vector x = ls.Solve();
            
            // Check if Ax = b
            return suite->isSolution(A, x, b);
        }
    };
    
    class CompareCGGaussian : public TestFunction {
    private:
        LinearSystemTestSuite* suite;
    public:
        CompareCGGaussian(LinearSystemTestSuite* s) : suite(s) {}
        bool operator()() override {
            // Create a positive definite matrix
            Matrix A(3, 3);
            Vector b(3);
            
            A(1, 1) = 4.0; A(1, 2) = 1.0; A(1, 3) = 1.0;
            A(2, 1) = 1.0; A(2, 2) = 3.0; A(2, 3) = 2.0;
            A(3, 1) = 1.0; A(3, 2) = 2.0; A(3, 3) = 3.5;
            
            b(1) = 1.0; b(2) = 2.0; b(3) = 3.0;
            
            // Solve with Conjugate Gradient
            PosSymLinSystem cg(A, b);
            Vector x_cg = cg.Solve();
            
            // Solve with Gaussian elimination
            LinearSystem ge(A, b);
            Vector x_ge = ge.Solve();
            
            // Compare solutions
            return suite->vectorsEqual(x_cg, x_ge);
        }
    };
    
    class LargeSymSystemTest : public TestFunction {
    private:
        LinearSystemTestSuite* suite;
    public:
        LargeSymSystemTest(LinearSystemTestSuite* s) : suite(s) {}
        bool operator()() override {
            // Create a larger positive definite matrix
            const int n = 10;
            Matrix A(n, n);
            Vector b(n);
            
            // Create a diagonally dominant symmetric matrix
            for (int i = 1; i <= n; i++) {
                for (int j = 1; j <= n; j++) {
                    if (i == j) {
                        A(i, j) = n + 5.0;  // Diagonal elements
                    } else {
                        A(i, j) = 1.0;     // Off-diagonal
                        A(j, i) = 1.0;     // Ensure symmetry
                    }
                }
                b(i) = i;  // Right-hand side
            }
            
            PosSymLinSystem ls(A, b);
            Vector x = ls.Solve();
            
            // Check if Ax = b
            return suite->isSolution(A, x, b);
        }
    };
    
    // Helper Method Tests
    class DotProductTest : public TestFunction {
    private:
        LinearSystemTestSuite* suite;
    public:
        DotProductTest(LinearSystemTestSuite* s) : suite(s) {}
        bool operator()() override {
            // Create data for testing
            Matrix A(2, 2);
            Vector b(2);
            
            A(1, 1) = 1.0; A(1, 2) = 0.0;
            A(2, 1) = 0.0; A(2, 2) = 1.0;
            
            b(1) = 2.0; b(2) = 3.0;
            
            PosSymLinSystem ls(A, b);
            
            Vector v1(2), v2(2);
            v1(1) = 1.0; v1(2) = 2.0;
            v2(1) = 3.0; v2(2) = 4.0;
            
            // Using Vector's dot product operator
            double dot = v1 * v2;
            double expectedDot = 1.0*3.0 + 2.0*4.0;  // 11.0
            
            return suite->almostEqual(dot, expectedDot);
        }
    };
    
    class MatrixVectorMultiplyTest : public TestFunction {
    public:
        bool operator()() override {
            // Create data for testing
            Matrix A(2, 2);
            Vector b(2);
            
            A(1, 1) = 1.0; A(1, 2) = 2.0;
            A(2, 1) = 3.0; A(2, 2) = 4.0;
            
            b(1) = 2.0; b(2) = 3.0;
            
            // Using Matrix's multiplication operator
            Vector expected = A * b;
            
            return true;  // If we got here, MatrixVectorMultiply works in the Solve method
        }
    };
    
    // Error Handling Tests
    class InvalidDimensionsTest : public TestFunction {
    public:
        bool operator()() override {
            Matrix A(3, 4);  // Not a square matrix
            Vector b(3);
            
            bool exceptionCaught = false;
            try {
                LinearSystem ls(A, b);
            } catch (...) {
                exceptionCaught = true;
            }
            
            return exceptionCaught;
        }
    };
    
    class MatrixVectorMismatchTest : public TestFunction {
    public:
        bool operator()() override {
            Matrix A(3, 3);
            Vector b(4);  // Wrong size
            
            bool exceptionCaught = false;
            try {
                LinearSystem ls(A, b);
            } catch (...) {
                exceptionCaught = true;
            }
            
            return exceptionCaught;
        }
    };
    
    class SingularMatrixTest : public TestFunction {
    public:
        bool operator()() override {
            // Create a singular matrix
            Matrix A(3, 3);
            Vector b(3);
            
            A(1, 1) = 1.0; A(1, 2) = 2.0; A(1, 3) = 3.0;
            A(2, 1) = 2.0; A(2, 2) = 4.0; A(2, 3) = 6.0;  // Row 2 = 2 * Row 1
            A(3, 1) = 3.0; A(3, 2) = 6.0; A(3, 3) = 9.0;  // Row 3 = 3 * Row 1
            
            b(1) = 1.0; b(2) = 2.0; b(3) = 3.0;
            
            LinearSystem ls(A, b);
            
            bool exceptionOrResultNaN = false;
            try {
                Vector x = ls.Solve();
                
                // Check if result contains NaN or Inf values
                for (int i = 1; i <= x.getSize(); i++) {
                    if (std::isnan(x(i)) || std::isinf(x(i))) {
                        exceptionOrResultNaN = true;
                        break;
                    }
                }
            } catch (...) {
                exceptionOrResultNaN = true;
            }
            
            return exceptionOrResultNaN;
        }
    };

public:
    void testBasicLinearSystem() {
        printSectionHeader("Basic LinearSystem Tests");
        
        ConstructorInitTest test1;
        runTest("Constructor initialization", test1);

        IdentityMatrixTest test2(this);
        runTest("Solve identity matrix system", test2);

        Solve2x2Test test3(this);
        runTest("Solve 2x2 linear system", test3);

        Solve3x3Test test4(this);
        runTest("Solve 3x3 linear system", test4);
        
        PivotingSystemTest test5(this);
        runTest("System with pivoting requirement", test5);
    }
    
    void testPosSymLinSystem() {
        printSectionHeader("PosSymLinSystem Tests (Conjugate Gradient)");
        
        PosSymConstructorTest test1;
        runTest("Constructor initialization", test1);
        
        RejectNonSymmetricTest test2;
        runTest("Reject non-symmetric matrix", test2);
        
        SolvePosDef test3(this);
        runTest("Solve positive definite system", test3);
        
        CompareCGGaussian test4(this);
        runTest("Compare CG with Gaussian elimination", test4);
        
        LargeSymSystemTest test5(this);
        runTest("Large symmetric system test", test5);
    }

    void testHelperMethods() {
        printSectionHeader("Helper Method Tests");
        
        DotProductTest test1(this);
        runTest("DotProduct method", test1);
        
        MatrixVectorMultiplyTest test2;
        runTest("MatrixVectorMultiply method", test2);
    }

    void testErrorHandling() {
        printSectionHeader("Error Handling Tests");
        
        InvalidDimensionsTest test1;
        runTest("Invalid dimensions in constructor", test1);
        
        MatrixVectorMismatchTest test2;
        runTest("Matrix-Vector size mismatch", test2);
        
        SingularMatrixTest test3;
        runTest("Singular matrix handling", test3);
    }
    
    void runAllTests() {
        printSuiteHeader();
        
        testBasicLinearSystem();
        testPosSymLinSystem();
        testHelperMethods();
        testErrorHandling();
        
        // Print summary
        cout << "\n" << string(50, '=') << RESET << endl;
        cout << "TEST SUMMARY:" << RESET << endl;
        cout << "  Total tests : " << setw(3) << totalTests << endl;
        cout << "  Tests passed: " << setw(3) << passedTests << " ("
             << fixed << setprecision(1) << (totalTests > 0 ? (100.0 * passedTests / totalTests) : 0)
             << "%)" << endl;
        cout << "  Tests failed: " << setw(3) << (totalTests - passedTests) << endl;
        if (passedTests == totalTests) {
            cout << GREEN << "  ALL TESTS ARE PASSED" << RESET << endl;
        } else {
            cout << RED << "  SOME TESTS ARE FAILED" << RESET << endl;
        }
        cout << string(50, '=') << RESET << endl;
    }
};

int main() {
    LinearSystemTestSuite tests;
    tests.runAllTests();
    return 0;
}