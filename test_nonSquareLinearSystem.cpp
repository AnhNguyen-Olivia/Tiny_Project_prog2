// filepath: c:\Users\Oath\OneDrive - student.vgu.edu.vn\Apps\Documents\GitHub\Tiny_Project_prog2\test_nonSquareLinearSystem.cpp
#include "nonSquareLinearSystem.h"
#include "matrix.h"
#include "vector.h"

#include <iostream>
#include <cassert>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>

using namespace std;

class NonSquareLinearSystemTestSuite {
private:
    int totalTests = 0;
    int passedTests = 0;
    const double EPSILON = 1e-6;  // For floating point comparisons

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

    // Check if Ax approximates b for over-determined systems
    bool isGoodSolution(const Matrix& A, const Vector& x, const Vector& b) const {
        Vector residual = A * x - b;
        double residualNorm = 0.0;
        for (int i = 1; i <= residual.getSize(); i++) {
            residualNorm += residual(i) * residual(i);
        }
        residualNorm = sqrt(residualNorm);
        return residualNorm < EPSILON * 10;  // Allow slightly larger error for least squares
    }

    void runTest(const string& testName, function<bool()> testFunction) {
        const int RESULT_COL = 62; // Column where PASS/FAIL/WARN starts
        totalTests++;

        // Print test name
        cout << "  " << testName;

        // Calculate and print dots so that result always starts at RESULT_COL
        int dots = RESULT_COL - 2 - static_cast<int>(testName.length());
        if (dots < 3) dots = 3;
        cout << string(dots, '.');

        cout << " ";

        try {
            bool result = testFunction();
            if (result) {
                passedTests++;
                cout << GREEN << "PASS" << RESET << endl;
            } else {
                cout << RED << "FAIL" << RESET << endl;
            }
        } catch (const exception& e) {
            cout << YELLOW << "WARN" << RESET << " (" << e.what() << ")" << endl;
        } catch (...) {
            cout << RED << "FAIL" << RESET << " (Unknown exception)" << endl;
        }
    }

    void printSectionHeader(const string& sectionName) {
        const int WIDTH = 70;
        string centered = sectionName;
        int pad = (WIDTH - centered.length()) / 2;
        cout << "\n" << string(WIDTH, '=') << endl;
        cout << string(pad, ' ') << centered << endl;
        cout << string(WIDTH, '=') << RESET << endl;
    }

public:
    void testConstructor() {
        printSectionHeader("Constructor Tests");
        
        runTest("Constructor with matching dimensions", [&]() {
            Matrix A(3, 2);  // 3x2 matrix (over-determined)
            Vector b(3);     // 3-element vector
            
            // Just testing if construction succeeds without assertions
            try {
                NonSquareLinearSystem nsls(A, b);
                return true;
            } catch (...) {
                return false;
            }
        });
        
        runTest("Constructor with mismatched dimensions", [&]() {
            Matrix A(3, 2);  // 3x2 matrix
            Vector b(4);     // 4-element vector (mismatch)
            
            // Should throw an assertion error
            bool assertionTriggered = false;
            try {
                NonSquareLinearSystem nsls(A, b);
            } catch (...) {
                assertionTriggered = true;
            }
            return assertionTriggered;
        });
    }

    void testOverdeterminedSystem() {
        printSectionHeader("Over-determined System Tests");
        
        runTest("Simple over-determined system with PseudoInverse", [&]() {
            // Create a simple over-determined system (more equations than unknowns)
            // 2x + y = 1
            // x + y = 0
            // x - y = 0
            Matrix A(3, 2);
            A(1, 1) = 2.0; A(1, 2) = 1.0;
            A(2, 1) = 1.0; A(2, 2) = 1.0;
            A(3, 1) = 1.0; A(3, 2) = -1.0;
            
            Vector b(3);
            b(1) = 1.0; b(2) = 0.0; b(3) = 0.0;
            
            NonSquareLinearSystem nsls(A, b);
            Vector x = nsls.SolveWithPseudoInverse();
            
            // Expected least-squares solution
            return isGoodSolution(A, x, b);
        });
        
        runTest("Real-world over-determined system with PseudoInverse", [&]() {
            // Create a realistic over-determined system
            Matrix A(5, 3);
            A(1, 1) = 1.0; A(1, 2) = 0.5; A(1, 3) = 0.2;
            A(2, 1) = 0.5; A(2, 2) = 1.0; A(2, 3) = 0.5;
            A(3, 1) = 0.2; A(3, 2) = 0.5; A(3, 3) = 1.0;
            A(4, 1) = 0.1; A(4, 2) = 0.2; A(4, 3) = 0.7;
            A(5, 1) = 0.8; A(5, 2) = 0.1; A(5, 3) = 0.3;
            
            Vector b(5);
            b(1) = 1.0; b(2) = 1.5; b(3) = 2.0; b(4) = 1.2; b(5) = 0.8;
            
            NonSquareLinearSystem nsls(A, b);
            Vector x = nsls.SolveWithPseudoInverse();
            
            return isGoodSolution(A, x, b);
        });
        
        runTest("Simple over-determined system with Tikhonov", [&]() {
            // Same system as above but solved with Tikhonov regularization
            Matrix A(3, 2);
            A(1, 1) = 2.0; A(1, 2) = 1.0;
            A(2, 1) = 1.0; A(2, 2) = 1.0;
            A(3, 1) = 1.0; A(3, 2) = -1.0;
            
            Vector b(3);
            b(1) = 1.0; b(2) = 0.0; b(3) = 0.0;
            
            NonSquareLinearSystem nsls(A, b);
            Vector x = nsls.SolveWithTikhonov(0.01); // Small regularization parameter
            
            return isGoodSolution(A, x, b);
        });
        
        runTest("Real-world over-determined system with Tikhonov", [&]() {
            // Same realistic system solved with Tikhonov regularization
            Matrix A(5, 3);
            A(1, 1) = 1.0; A(1, 2) = 0.5; A(1, 3) = 0.2;
            A(2, 1) = 0.5; A(2, 2) = 1.0; A(2, 3) = 0.5;
            A(3, 1) = 0.2; A(3, 2) = 0.5; A(3, 3) = 1.0;
            A(4, 1) = 0.1; A(4, 2) = 0.2; A(4, 3) = 0.7;
            A(5, 1) = 0.8; A(5, 2) = 0.1; A(5, 3) = 0.3;
            
            Vector b(5);
            b(1) = 1.0; b(2) = 1.5; b(3) = 2.0; b(4) = 1.2; b(5) = 0.8;
            
            NonSquareLinearSystem nsls(A, b);
            Vector x = nsls.SolveWithTikhonov(0.01); // Small regularization parameter
            
            return isGoodSolution(A, x, b);
        });
    }

    void testUnderdeterminedSystem() {
        printSectionHeader("Under-determined System Tests");
        
        runTest("Simple under-determined system with PseudoInverse", [&]() {
            // Create a simple under-determined system (fewer equations than unknowns)
            // x + y + z = 1
            // x - y + 2z = 2
            Matrix A(2, 3);
            A(1, 1) = 1.0; A(1, 2) = 1.0; A(1, 3) = 1.0;
            A(2, 1) = 1.0; A(2, 2) = -1.0; A(2, 3) = 2.0;
            
            Vector b(2);
            b(1) = 1.0; b(2) = 2.0;
            
            NonSquareLinearSystem nsls(A, b);
            Vector x = nsls.SolveWithPseudoInverse();
            
            // Check if Ax = b exactly (since this is under-determined)
            Vector Ax = A * x;
            return vectorsEqual(Ax, b);
        });
        
        runTest("Real-world under-determined system with PseudoInverse", [&]() {
            // More complex under-determined system
            Matrix A(2, 4);
            A(1, 1) = 1.0; A(1, 2) = 2.0; A(1, 3) = 3.0; A(1, 4) = 4.0;
            A(2, 1) = 5.0; A(2, 2) = 6.0; A(2, 3) = 7.0; A(2, 4) = 8.0;
            
            Vector b(2);
            b(1) = 10.0; b(2) = 26.0;
            
            NonSquareLinearSystem nsls(A, b);
            Vector x = nsls.SolveWithPseudoInverse();
            
            // Check if Ax = b exactly (since this is under-determined)
            Vector Ax = A * x;
            return vectorsEqual(Ax, b);
        });
        
        runTest("Simple under-determined system with Tikhonov", [&]() {
            // Same simple under-determined system but solved with Tikhonov
            Matrix A(2, 3);
            A(1, 1) = 1.0; A(1, 2) = 1.0; A(1, 3) = 1.0;
            A(2, 1) = 1.0; A(2, 2) = -1.0; A(2, 3) = 2.0;
            
            Vector b(2);
            b(1) = 1.0; b(2) = 2.0;
            
            NonSquareLinearSystem nsls(A, b);
            Vector x = nsls.SolveWithTikhonov(0.01); // Small regularization parameter
            
            // For Tikhonov, Ax may not exactly equal b, but should be close
            Vector Ax = A * x;
            double residualNorm = 0.0;
            for (int i = 1; i <= Ax.getSize(); i++) {
                residualNorm += (Ax(i) - b(i)) * (Ax(i) - b(i));
            }
            return sqrt(residualNorm) < EPSILON * 10;
        });
        
        runTest("Real-world under-determined system with Tikhonov", [&]() {
            // More complex under-determined system with Tikhonov
            Matrix A(2, 4);
            A(1, 1) = 1.0; A(1, 2) = 2.0; A(1, 3) = 3.0; A(1, 4) = 4.0;
            A(2, 1) = 5.0; A(2, 2) = 6.0; A(2, 3) = 7.0; A(2, 4) = 8.0;
            
            Vector b(2);
            b(1) = 10.0; b(2) = 26.0;
            
            NonSquareLinearSystem nsls(A, b);
            Vector x = nsls.SolveWithTikhonov(0.01); // Small regularization parameter
            
            // For Tikhonov, Ax may not exactly equal b, but should be close
            Vector Ax = A * x;
            double residualNorm = 0.0;
            for (int i = 1; i <= Ax.getSize(); i++) {
                residualNorm += (Ax(i) - b(i)) * (Ax(i) - b(i));
            }
            return sqrt(residualNorm) < EPSILON * 10;
        });
    }
    
    void testComparisonMethods() {
        printSectionHeader("Method Comparison Tests");
        
        runTest("Compare PseudoInverse and Tikhonov for over-determined", [&]() {
            Matrix A(4, 2);
            A(1, 1) = 1.0; A(1, 2) = 0.5;
            A(2, 1) = 0.5; A(2, 2) = 1.0;
            A(3, 1) = 0.2; A(3, 2) = 0.8;
            A(4, 1) = 0.8; A(4, 2) = 0.2;
            
            Vector b(4);
            b(1) = 1.0; b(2) = 1.5; b(3) = 1.0; b(4) = 1.0;
            
            NonSquareLinearSystem nsls(A, b);
            Vector x1 = nsls.SolveWithPseudoInverse();
            Vector x2 = nsls.SolveWithTikhonov(0.0001); // Very small lambda should approach pseudo-inverse
            
            // Solutions should be similar for small lambda
            return vectorsEqual(x1, x2);
        });
        
        runTest("Compare PseudoInverse and Tikhonov for under-determined", [&]() {
            Matrix A(2, 4);
            A(1, 1) = 1.0; A(1, 2) = 0.5; A(1, 3) = 0.2; A(1, 4) = 0.8;
            A(2, 1) = 0.5; A(2, 2) = 1.0; A(2, 3) = 0.8; A(2, 4) = 0.2;
            
            Vector b(2);
            b(1) = 1.0; b(2) = 1.5;
            
            NonSquareLinearSystem nsls(A, b);
            Vector x1 = nsls.SolveWithPseudoInverse();
            Vector x2 = nsls.SolveWithTikhonov(0.0001); // Very small lambda should approach pseudo-inverse
            
            // Solutions should be similar for small lambda
            return vectorsEqual(x1, x2);
        });
    }
    
    void testRegularizationEffect() {
        printSectionHeader("Regularization Effect Tests");
        
        runTest("Tikhonov solution norm decreases with higher lambda", [&]() {
            Matrix A(3, 4);
            A(1, 1) = 1.0; A(1, 2) = 2.0; A(1, 3) = 3.0; A(1, 4) = 4.0;
            A(2, 1) = 5.0; A(2, 2) = 6.0; A(2, 3) = 7.0; A(2, 4) = 8.0;
            A(3, 1) = 9.0; A(3, 2) = 10.0; A(3, 3) = 11.0; A(3, 4) = 12.0;
            
            Vector b(3);
            b(1) = 1.0; b(2) = 2.0; b(3) = 3.0;
            
            NonSquareLinearSystem nsls(A, b);
            Vector x1 = nsls.SolveWithTikhonov(0.01);   // Small lambda
            Vector x2 = nsls.SolveWithTikhonov(10.0);   // Large lambda
            
            // Compute norms of both solutions
            double norm1 = 0.0, norm2 = 0.0;
            for (int i = 1; i <= x1.getSize(); i++) {
                norm1 += x1(i) * x1(i);
                norm2 += x2(i) * x2(i);
            }
            norm1 = sqrt(norm1);
            norm2 = sqrt(norm2);
            
            // Solution norm should decrease as lambda increases
            return norm2 < norm1;
        });
    }
    
    void runAllTests() {
        cout << "\n" << string(70, '=') << endl;
        cout << "           NON-SQUARE LINEAR SYSTEM TEST SUITE" << endl;
        cout << string(70, '=') << endl;

        testConstructor();
        testOverdeterminedSystem();
        testUnderdeterminedSystem();
        testComparisonMethods();
        testRegularizationEffect();

        // Print summary
        cout << "\n" << string(70, '=') << endl;
        cout << "TEST SUMMARY:" << endl;
        cout << "  Total tests : " << setw(3) << totalTests << endl;
        cout << "  Tests passed: " << setw(3) << passedTests << " ("
             << fixed << setprecision(1) << (totalTests > 0 ? (100.0 * passedTests / totalTests) : 0)
             << "%)" << endl;
        cout << "  Tests failed: " << setw(3) << (totalTests - passedTests) << endl;
        
        if (passedTests == totalTests) {
            cout << GREEN << "  All tests passed!" << RESET << endl;
        } else {
            cout << RED << "  Some tests failed!" << RESET << endl;
        }
        
        cout << string(70, '=') << endl;
    }
};

int main() {
    NonSquareLinearSystemTestSuite tests;
    tests.runAllTests();
    return 0;
}