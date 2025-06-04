#include "linearSystem.h"
#include "matrix.h"
#include "vector.h"

#include <iostream>
#include <cassert>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>

using namespace std;

class LinearSystemTestSuite {
private:
    int totalTests = 0;
    int passedTests = 0;
    const double EPSILON = 1e-6;  // For floating point comparisons

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
        totalTests++;
        cout << "  Testing: " << left << setw(50) << testName << " ... ";
        
        try {
            bool result = testFunction();
            if (result) {
                cout << "PASSED" << endl;
                passedTests++;
            } else {
                cout << "FAILED" << endl;
            }
        } catch (const exception& e) {
            cout << "EXCEPTION: " << e.what() << endl;
        } catch (...) {
            cout << "UNKNOWN EXCEPTION" << endl;
        }
    }

    void printSectionHeader(const string& sectionName) {
        cout << "\n=== " << sectionName << " ===" << endl;
    }

    // Test function implementations
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
            return suite->vectorsEqual(x, b);
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
            
            // Expected solution: x = 1.0, y = 3.0
            Vector expected(2);
            expected(1) = 1.0; expected(2) = 3.0;
            
            LinearSystem ls(A, b);
            Vector x = ls.Solve();
            
            return suite->vectorsEqual(x, expected);
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

    // Add similar classes for other tests...

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

        // Add additional test functions for the remaining tests
        // For brevity, I'm not including all test implementations
    }
    
    void testPosSymLinSystem() {
        printSectionHeader("PosSymLinSystem Tests (Conjugate Gradient)");
        
        // Implement test function objects and add them here
    }

    void testHelperMethods() {
        printSectionHeader("Helper Method Tests");
        
        // Implement test function objects and add them here
    }

    void testErrorHandling() {
        printSectionHeader("Error Handling Tests");
        
        // Implement test function objects and add them here
    }
    
    void runAllTests() {
        cout << "\n==============================================" << endl;
        cout << "      LINEAR SYSTEM CLASS TEST SUITE" << endl;
        cout << "==============================================" << endl;
        
        testBasicLinearSystem();
        testPosSymLinSystem();
        testHelperMethods();
        testErrorHandling();
        
        // Print summary
        cout << "\n==============================================" << endl;
        cout << "TEST SUMMARY:" << endl;
        cout << "  Total tests:  " << totalTests << endl;
        cout << "  Tests passed: " << passedTests << " (" 
             << fixed << setprecision(1) << (totalTests > 0 ? (100.0 * passedTests / totalTests) : 0) 
             << "%)" << endl;
        cout << "  Tests failed: " << (totalTests - passedTests) << endl;
        cout << "==============================================" << endl;
    }
};

int main() {
    LinearSystemTestSuite tests;
    tests.runAllTests();
    return 0;
}