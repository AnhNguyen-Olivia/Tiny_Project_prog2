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

class MatrixTestSuite {
private:
    int totalTests = 0;
    int passedTests = 0;
    int testCounter = 1; // For test IDs
    const double EPSILON = 1e-10;  // For floating point comparisons

    // ANSI color codes for output
    const string GREEN = "\033[32m";
    const string RED = "\033[31m";
    const string YELLOW = "\033[33m";
    const string RESET = "\033[0m";

    bool almostEqual(double a, double b) const {
        return fabs(a - b) < EPSILON;
    }

    bool matricesEqual(const Matrix& m1, const Matrix& m2) const {
        if (m1.GetNumRows() != m2.GetNumRows() || m1.GetNumCols() != m2.GetNumCols()) return false;
        for (int i = 1; i <= m1.GetNumRows(); i++) {
            for (int j = 1; j <= m1.GetNumCols(); j++) {
                if (!almostEqual(m1(i, j), m2(i, j))) return false;
            }
        }
        return true;
    }

    void runTest(const string& testName, function<bool()> testFunction) {
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
        cout << "Test Suite: MATRIX CLASS " << VERSION << endl;
        cout << "Run date: " << timebuf << endl;
        cout << string(50, '=') << RESET << endl;
    }

    void printSectionHeader(const string& sectionName) {
        cout << "\n=== " << sectionName << " ===" << endl;
    }

public:
    void testConstructors() {
        printSectionHeader("Constructor Tests");
        
        runTest("Size constructor initialization", [&]() {
            Matrix m(3, 4);
            bool allZeros = true;
            for (int i = 1; i <= 3; i++) {
                for (int j = 1; j <= 4; j++) {
                    if (!almostEqual(m(i, j), 0.0)) {
                        allZeros = false;
                        break;
                    }
                }
            }
            return allZeros && m.GetNumRows() == 3 && m.GetNumCols() == 4;
        });
        
        runTest("Copy constructor", [&]() {
            Matrix m1(2, 3);
            m1(1, 1) = 1.5;
            m1(1, 2) = 2.5;
            m1(2, 3) = 3.5;
            
            Matrix m2(m1);
            return matricesEqual(m1, m2);
        });
        
        runTest("Identity matrix static constructor", [&]() {
            Matrix m = Matrix::IdentityMatrix(3);
            bool validIdentity = true;
            
            for (int i = 1; i <= 3; i++) {
                for (int j = 1; j <= 3; j++) {
                    if (i == j) {
                        if (!almostEqual(m(i, j), 1.0)) validIdentity = false;
                    } else {
                        if (!almostEqual(m(i, j), 0.0)) validIdentity = false;
                    }
                }
            }
            
            return validIdentity;
        });
    }

    void testAccessors() {
        printSectionHeader("Accessor Tests");
        
        runTest("GetNumRows()", [&]() {
            Matrix m(5, 3);
            return m.GetNumRows() == 5;
        });
        
        runTest("GetNumCols()", [&]() {
            Matrix m(5, 3);
            return m.GetNumCols() == 3;
        });
        
        runTest("Operator(i,j) access", [&]() {
            Matrix m(2, 2);
            m(1, 1) = 1.0;
            m(1, 2) = 2.0;
            m(2, 1) = 3.0;
            m(2, 2) = 4.0;
            
            return almostEqual(m(1, 1), 1.0) && 
                   almostEqual(m(1, 2), 2.0) && 
                   almostEqual(m(2, 1), 3.0) && 
                   almostEqual(m(2, 2), 4.0);
        });
        
        runTest("Const operator(i,j) access", [&]() {
            Matrix m(2, 2);
            m(1, 1) = 1.0;
            m(1, 2) = 2.0;
            m(2, 1) = 3.0;
            m(2, 2) = 4.0;
            
            const Matrix& constM = m;
            return almostEqual(constM(1, 1), 1.0) && 
                   almostEqual(constM(1, 2), 2.0) && 
                   almostEqual(constM(2, 1), 3.0) && 
                   almostEqual(constM(2, 2), 4.0);
        });
    }
    
    void testAssignment() {
        printSectionHeader("Assignment Tests");
        
        runTest("Assignment operator", [&]() {
            Matrix m1(2, 3);
            m1(1, 1) = 1.0;
            m1(1, 2) = 2.0;
            m1(1, 3) = 3.0;
            m1(2, 1) = 4.0;
            m1(2, 2) = 5.0;
            m1(2, 3) = 6.0;
            
            Matrix m2(1, 1);  // Different size
            m2 = m1;
            
            return matricesEqual(m1, m2) && 
                   m2.GetNumRows() == 2 && 
                   m2.GetNumCols() == 3;
        });
        
        runTest("Self-assignment safety", [&]() {
            Matrix m1(2, 2);
            m1(1, 1) = 1.0;
            m1(1, 2) = 2.0;
            m1(2, 1) = 3.0;
            m1(2, 2) = 4.0;
            
            Matrix m2(m1);
            m1 = m1;  // Self-assignment
            
            return matricesEqual(m1, m2);
        });
    }
    
    void testBinaryOperators() {
        printSectionHeader("Binary Operator Tests");
        
        runTest("Addition operator (+)", [&]() {
            Matrix m1(2, 2);
            m1(1, 1) = 1.0;
            m1(1, 2) = 2.0;
            m1(2, 1) = 3.0;
            m1(2, 2) = 4.0;
            
            Matrix m2(2, 2);
            m2(1, 1) = 5.0;
            m2(1, 2) = 6.0;
            m2(2, 1) = 7.0;
            m2(2, 2) = 8.0;
            
            Matrix sum = m1 + m2;
            
            return almostEqual(sum(1, 1), 6.0) && 
                   almostEqual(sum(1, 2), 8.0) && 
                   almostEqual(sum(2, 1), 10.0) && 
                   almostEqual(sum(2, 2), 12.0);
        });
        
        runTest("Subtraction operator (-)", [&]() {
            Matrix m1(2, 2);
            m1(1, 1) = 5.0;
            m1(1, 2) = 6.0;
            m1(2, 1) = 7.0;
            m1(2, 2) = 8.0;
            
            Matrix m2(2, 2);
            m2(1, 1) = 1.0;
            m2(1, 2) = 2.0;
            m2(2, 1) = 3.0;
            m2(2, 2) = 4.0;
            
            Matrix diff = m1 - m2;
            
            return almostEqual(diff(1, 1), 4.0) && 
                   almostEqual(diff(1, 2), 4.0) && 
                   almostEqual(diff(2, 1), 4.0) && 
                   almostEqual(diff(2, 2), 4.0);
        });
        
        runTest("Scalar multiplication (*)", [&]() {
            Matrix m(2, 2);
            m(1, 1) = 1.0;
            m(1, 2) = 2.0;
            m(2, 1) = 3.0;
            m(2, 2) = 4.0;
            
            Matrix result = m * 2.5;
            
            return almostEqual(result(1, 1), 2.5) && 
                   almostEqual(result(1, 2), 5.0) && 
                   almostEqual(result(2, 1), 7.5) && 
                   almostEqual(result(2, 2), 10.0);
        });
        
        runTest("Vector multiplication (*)", [&]() {
            Matrix m(2, 3);
            m(1, 1) = 1.0;
            m(1, 2) = 2.0;
            m(1, 3) = 3.0;
            m(2, 1) = 4.0;
            m(2, 2) = 5.0;
            m(2, 3) = 6.0;
            
            Vector v(3);
            v(1) = 7.0;
            v(2) = 8.0;
            v(3) = 9.0;
            
            Vector result = m * v;
            
            return result.getSize() == 2 &&
                   almostEqual(result(1), 1.0*7.0 + 2.0*8.0 + 3.0*9.0) && 
                   almostEqual(result(2), 4.0*7.0 + 5.0*8.0 + 6.0*9.0);
        });
        
        runTest("Matrix multiplication (*)", [&]() {
            Matrix m1(2, 3);
            m1(1, 1) = 1.0;
            m1(1, 2) = 2.0;
            m1(1, 3) = 3.0;
            m1(2, 1) = 4.0;
            m1(2, 2) = 5.0;
            m1(2, 3) = 6.0;
            
            Matrix m2(3, 2);
            m2(1, 1) = 7.0;
            m2(1, 2) = 8.0;
            m2(2, 1) = 9.0;
            m2(2, 2) = 10.0;
            m2(3, 1) = 11.0;
            m2(3, 2) = 12.0;
            
            Matrix result = m1 * m2;
            
            return result.GetNumRows() == 2 && 
                   result.GetNumCols() == 2 &&
                   almostEqual(result(1, 1), 1.0*7.0 + 2.0*9.0 + 3.0*11.0) && 
                   almostEqual(result(1, 2), 1.0*8.0 + 2.0*10.0 + 3.0*12.0) &&
                   almostEqual(result(2, 1), 4.0*7.0 + 5.0*9.0 + 6.0*11.0) && 
                   almostEqual(result(2, 2), 4.0*8.0 + 5.0*10.0 + 6.0*12.0);
        });
    }
    
    void testUnaryOperators() {
        printSectionHeader("Unary Operator Tests");
        
        runTest("Negation operator (-)", [&]() {
            Matrix m(2, 2);
            m(1, 1) = 1.0;
            m(1, 2) = 2.0;
            m(2, 1) = 3.0;
            m(2, 2) = 4.0;
            
            Matrix negated = -m;
            
            return almostEqual(negated(1, 1), -1.0) && 
                   almostEqual(negated(1, 2), -2.0) && 
                   almostEqual(negated(2, 1), -3.0) && 
                   almostEqual(negated(2, 2), -4.0);
        });
    }
    
    void testUtilityFunctions() {
        printSectionHeader("Utility Function Tests");
        
        runTest("Transpose()", [&]() {
            Matrix m(2, 3);
            m(1, 1) = 1.0;
            m(1, 2) = 2.0;
            m(1, 3) = 3.0;
            m(2, 1) = 4.0;
            m(2, 2) = 5.0;
            m(2, 3) = 6.0;
            
            Matrix t = m.Transpose();
            
            return t.GetNumRows() == 3 && 
                   t.GetNumCols() == 2 &&
                   almostEqual(t(1, 1), 1.0) && 
                   almostEqual(t(1, 2), 4.0) && 
                   almostEqual(t(2, 1), 2.0) && 
                   almostEqual(t(2, 2), 5.0) &&
                   almostEqual(t(3, 1), 3.0) && 
                   almostEqual(t(3, 2), 6.0);
        });
        
        runTest("Determinant() for 1x1", [&]() {
            Matrix m(1, 1);
            m(1, 1) = 5.0;
            return almostEqual(m.Determinant(), 5.0);
        });
        
        runTest("Determinant() for 2x2", [&]() {
            Matrix m(2, 2);
            m(1, 1) = 1.0;
            m(1, 2) = 2.0;
            m(2, 1) = 3.0;
            m(2, 2) = 4.0;
            // det = 1*4 - 2*3 = 4 - 6 = -2
            return almostEqual(m.Determinant(), -2.0);
        });
        
        runTest("Determinant() for 3x3", [&]() {
            Matrix m(3, 3);
            m(1, 1) = 1.0; m(1, 2) = 2.0; m(1, 3) = 3.0;
            m(2, 1) = 4.0; m(2, 2) = 5.0; m(2, 3) = 6.0;
            m(3, 1) = 7.0; m(3, 2) = 8.0; m(3, 3) = 9.0;
            // det = 0 for this singular matrix
            return almostEqual(m.Determinant(), 0.0);
        });
        
        runTest("Inverse() for 2x2", [&]() {
            Matrix m(2, 2);
            m(1, 1) = 4.0;
            m(1, 2) = 7.0;
            m(2, 1) = 2.0;
            m(2, 2) = 6.0;
            // det = 4*6 - 7*2 = 24 - 14 = 10
            // inverse = [[6, -7], [-2, 4]] / 10
            
            Matrix inv = m.Inverse();
            
            return almostEqual(inv(1, 1), 6.0/10) && 
                   almostEqual(inv(1, 2), -7.0/10) && 
                   almostEqual(inv(2, 1), -2.0/10) && 
                   almostEqual(inv(2, 2), 4.0/10);
        });
        
        runTest("Square() with square matrix", [&]() {
            Matrix m(3, 3);
            return m.Square() == true;
        });
        
        runTest("Square() with non-square matrix", [&]() {
            Matrix m(3, 4);
            return m.Square() == false;
        });
        
        runTest("Symmetric() with symmetric matrix", [&]() {
            Matrix m(3, 3);
            m(1, 1) = 1.0; m(1, 2) = 2.0; m(1, 3) = 3.0;
            m(2, 1) = 2.0; m(2, 2) = 4.0; m(2, 3) = 5.0;
            m(3, 1) = 3.0; m(3, 2) = 5.0; m(3, 3) = 6.0;
            
            return m.Symmetric() == true;
        });
        
        runTest("Symmetric() with non-symmetric matrix", [&]() {
            Matrix m(3, 3);
            m(1, 1) = 1.0; m(1, 2) = 2.0; m(1, 3) = 3.0;
            m(2, 1) = 4.0; m(2, 2) = 5.0; m(2, 3) = 6.0;
            m(3, 1) = 7.0; m(3, 2) = 8.0; m(3, 3) = 9.0;
            
            return m.Symmetric() == false;
        });
    }
    
    void testErrorHandling() {
        printSectionHeader("Error Handling Tests");
        
        runTest("Invalid size constructor", [&]() {
            bool exceptionCaught = false;
            try {
                Matrix m(0, 5);  // Should assert
            } catch (...) {
                exceptionCaught = true;
            }
            return exceptionCaught;
        });
        
        runTest("Invalid operator() access", [&]() {
            Matrix m(3, 3);
            bool exceptionCaught = false;
            try {
                double x = m(0, 2);  // Should assert, () is 1-based
            } catch (...) {
                exceptionCaught = true;
            }
            return exceptionCaught;
        });
        
        runTest("Matrix size mismatch in addition", [&]() {
            Matrix m1(2, 3), m2(3, 2);
            bool exceptionCaught = false;
            try {
                Matrix sum = m1 + m2;  // Different sizes should assert
            } catch (...) {
                exceptionCaught = true;
            }
            return exceptionCaught;
        });
        
        runTest("Matrix-Matrix multiplication dimension mismatch", [&]() {
            Matrix m1(2, 3), m2(4, 2);
            bool exceptionCaught = false;
            try {
                Matrix product = m1 * m2;  // Incompatible dimensions should assert
            } catch (...) {
                exceptionCaught = true;
            }
            return exceptionCaught;
        });
        
        runTest("Determinant of non-square matrix", [&]() {
            Matrix m(2, 3);
            bool exceptionCaught = false;
            try {
                double det = m.Determinant();  // Non-square matrix should assert
            } catch (...) {
                exceptionCaught = true;
            }
            return exceptionCaught;
        });
        
        runTest("Inverse of singular matrix", [&]() {
            Matrix m(2, 2);
            m(1, 1) = 1.0; m(1, 2) = 2.0;
            m(2, 1) = 2.0; m(2, 2) = 4.0;
            // Determinant is 1*4 - 2*2 = 0
            
            bool exceptionCaught = false;
            try {
                Matrix inv = m.Inverse();  // Singular matrix should assert
            } catch (...) {
                exceptionCaught = true;
            }
            return exceptionCaught;
        });
    }
    
    void runAllTests() {

        printSuiteHeader();

        testConstructors();
        testAccessors();
        testAssignment();
        testBinaryOperators();
        testUnaryOperators();
        testUtilityFunctions();
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
    MatrixTestSuite tests;
    tests.runAllTests();
    return 0;
}