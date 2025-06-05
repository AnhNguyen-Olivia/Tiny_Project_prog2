#include "vector.h"
#include <iostream>
#include <cassert>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>
#include <vector>
#include <chrono>
#include <ctime>

using namespace std;

class VectorTestSuite {
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

    bool vectorsEqual(const Vector& v1, const Vector& v2) const {
        if (v1.getSize() != v2.getSize()) return false;
        for (int i = 0; i < v1.getSize(); i++) {
            if (!almostEqual(v1[i], v2[i])) return false;
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
        tm local_tm;
        localtime_r(&now_c, &local_tm);
        char timebuf[32];
        strftime(timebuf, sizeof(timebuf), "%Y-%m-%d  %H:%M:%S", &local_tm);

        cout << "\n" << string(50, '=') << RESET << endl;
        cout << "Test Suite: VECTOR CLASS " << VERSION << endl;
        cout << "Run date: " << timebuf << endl;
        cout << string(50, '=') << RESET << endl;
    }

public:
    // Helper to print section headers in the test output
    void printSectionHeader(const string& sectionName) {
        cout << "\n=== " << sectionName << " ===" << endl;
    }

    void testConstructors() {
        this->printSectionHeader("Constructor Tests");
        
        runTest("Default constructor initialization", [&]() {
            Vector v(3);
            return almostEqual(v[0], 0.0) && almostEqual(v[1], 0.0) && almostEqual(v[2], 0.0);
            // Should be: Vector with [0.00, 0.00, 0.00]
        });
        
        runTest("Array constructor", [&]() {
            double arr[] = {1.5, 2.5, 3.5};
            Vector v(arr, 3);
            return almostEqual(v[0], 1.5) && almostEqual(v[1], 2.5) && almostEqual(v[2], 3.5);
            // Should be: Vector with [1.50, 2.50, 3.50]
        });
        
        runTest("Copy constructor", [&]() {
            double arr[] = {1.5, 2.5, 3.5};
            Vector v1(arr, 3);
            Vector v2(v1);
            return vectorsEqual(v1, v2);
            // Should be: Two identical vectors with [1.50, 2.50, 3.50]
        });
    }

    void testAccessors() {
        this->printSectionHeader("Accessor Tests");
        
        runTest("getSize()", [&]() {
            Vector v(5);
            return v.getSize() == 5;
            // Should be: Size = 5
        });
        
        runTest("getData() pointer validity", [&]() {
            Vector v(3);
            double* data = v.getData();
            return data != nullptr;
            // Should be: Valid non-null pointer
        });
        
        runTest("Bracket operator []", [&]() {
            double arr[] = {1.5, 2.5, 3.5};
            Vector v(arr, 3);
            return almostEqual(v[0], 1.5) && almostEqual(v[1], 2.5) && almostEqual(v[2], 3.5);
            // Should be: v[0]=1.50, v[1]=2.50, v[2]=3.50
        });
        
        runTest("Parentheses operator ()", [&]() {
            double arr[] = {1.5, 2.5, 3.5};
            Vector v(arr, 3);
            return almostEqual(v(1), 1.5) && almostEqual(v(2), 2.5) && almostEqual(v(3), 3.5);
            // Should be: v(1)=1.50, v(2)=2.50, v(3)=3.50
        });
        
        runTest("Element modification via []", [&]() {
            Vector v(3);
            v[0] = 1.5;
            v[1] = 2.5;
            v[2] = 3.5;
            return almostEqual(v[0], 1.5) && almostEqual(v[1], 2.5) && almostEqual(v[2], 3.5);
            // Should be: Vector with [1.50, 2.50, 3.50] after modification
        });
        
        runTest("Element modification via ()", [&]() {
            Vector v(3);
            v(1) = 1.5;
            v(2) = 2.5;
            v(3) = 3.5;
            return almostEqual(v[0], 1.5) && almostEqual(v[1], 2.5) && almostEqual(v[2], 3.5);
            // Should be: Vector with [1.50, 2.50, 3.50] after modification
        });
    }
    
    void testAssignment() {
        this->printSectionHeader("Assignment Tests");
        
        runTest("Assignment operator", [&]() {
            double arr[] = {1.5, 2.5, 3.5};
            Vector v1(arr, 3);
            Vector v2(2);  // Different size
            v2 = v1;
            return vectorsEqual(v1, v2) && v2.getSize() == 3;
            // Should be: v2 becomes identical to v1 with size 3 and values [1.50, 2.50, 3.50]
        });
        
        runTest("Self-assignment safety", [&]() {
            double arr[] = {1.5, 2.5, 3.5};
            Vector v1(arr, 3);
            Vector v2(v1);
            v1 = v1;  // Self-assignment
            return vectorsEqual(v1, v2);
            // Should be: v1 remains unchanged with [1.50, 2.50, 3.50]
        });
    }
    
    void testUnaryOperators() {
        this->printSectionHeader("Unary Operator Tests");
        
        runTest("Negation operator (-)", [&]() {
            double arr[] = {1.5, 2.5, 3.5};
            Vector v1(arr, 3);
            Vector v2 = -v1;
            return almostEqual(v2[0], -1.5) && almostEqual(v2[1], -2.5) && almostEqual(v2[2], -3.5);
            // Should be: v2 = [-1.50, -2.50, -3.50]
        });
        
        runTest("Pre-increment operator (++)", [&]() {
            double arr[] = {1.0, 2.0, 3.0};
            Vector v(arr, 3);
            Vector pre = ++v;
            return almostEqual(v[0], 2.0) && almostEqual(v[1], 3.0) && almostEqual(v[2], 4.0) &&
                   almostEqual(pre[0], 2.0) && almostEqual(pre[1], 3.0) && almostEqual(pre[2], 4.0);
            // Should be: v = [2.00, 3.00, 4.00], pre = [2.00, 3.00, 4.00]
        });
        
        runTest("Post-increment operator (++)", [&]() {
            double arr[] = {1.0, 2.0, 3.0};
            Vector v(arr, 3);
            Vector post = v++;
            return almostEqual(v[0], 2.0) && almostEqual(v[1], 3.0) && almostEqual(v[2], 4.0) &&
                   almostEqual(post[0], 1.0) && almostEqual(post[1], 2.0) && almostEqual(post[2], 3.0);
            // Should be: v = [2.00, 3.00, 4.00], post = [1.00, 2.00, 3.00]
        });
        
        runTest("Pre-decrement operator (--)", [&]() {
            double arr[] = {3.0, 4.0, 5.0};
            Vector v(arr, 3);
            Vector pre = --v;
            return almostEqual(v[0], 2.0) && almostEqual(v[1], 3.0) && almostEqual(v[2], 4.0) &&
                   almostEqual(pre[0], 2.0) && almostEqual(pre[1], 3.0) && almostEqual(pre[2], 4.0);
            // Should be: v = [2.00, 3.00, 4.00], pre = [2.00, 3.00, 4.00]
        });
        
        runTest("Post-decrement operator (--)", [&]() {
            double arr[] = {3.0, 4.0, 5.0};
            Vector v(arr, 3);
            Vector post = v--;
            return almostEqual(v[0], 2.0) && almostEqual(v[1], 3.0) && almostEqual(v[2], 4.0) &&
                   almostEqual(post[0], 3.0) && almostEqual(post[1], 4.0) && almostEqual(post[2], 5.0);
            // Should be: v = [2.00, 3.00, 4.00], post = [3.00, 4.00, 5.00]
        });
    }
    
    void testBinaryOperators() {
        this->printSectionHeader("Binary Operator Tests");
        
        runTest("Addition operator (+)", [&]() {
            double arr1[] = {1.0, 2.0, 3.0};
            double arr2[] = {4.0, 5.0, 6.0};
            Vector v1(arr1, 3);
            Vector v2(arr2, 3);
            Vector sum = v1 + v2;
            return almostEqual(sum[0], 5.0) && almostEqual(sum[1], 7.0) && almostEqual(sum[2], 9.0);
            // Should be: sum = [5.00, 7.00, 9.00]
        });
        
        runTest("Addition assignment operator (+=)", [&]() {
            double arr1[] = {1.0, 2.0, 3.0};
            double arr2[] = {4.0, 5.0, 6.0};
            Vector v1(arr1, 3);
            Vector v2(arr2, 3);
            v1 += v2;
            return almostEqual(v1[0], 5.0) && almostEqual(v1[1], 7.0) && almostEqual(v1[2], 9.0);
            // Should be: v1 = [5.00, 7.00, 9.00]
        });
        
        runTest("Subtraction operator (-)", [&]() {
            double arr1[] = {4.0, 5.0, 6.0};
            double arr2[] = {1.0, 2.0, 3.0};
            Vector v1(arr1, 3);
            Vector v2(arr2, 3);
            Vector diff = v1 - v2;
            return almostEqual(diff[0], 3.0) && almostEqual(diff[1], 3.0) && almostEqual(diff[2], 3.0);
            // Should be: diff = [3.00, 3.00, 3.00]
        });
        
        runTest("Subtraction assignment operator (-=)", [&]() {
            double arr1[] = {4.0, 5.0, 6.0};
            double arr2[] = {1.0, 2.0, 3.0};
            Vector v1(arr1, 3);
            Vector v2(arr2, 3);
            v1 -= v2;
            return almostEqual(v1[0], 3.0) && almostEqual(v1[1], 3.0) && almostEqual(v1[2], 3.0);
            // Should be: v1 = [3.00, 3.00, 3.00]
        });
        
        runTest("Scalar multiplication operator (*)", [&]() {
            double arr[] = {1.0, 2.0, 3.0};
            Vector v(arr, 3);
            Vector result = v * 2.5;
            return almostEqual(result[0], 2.5) && almostEqual(result[1], 5.0) && almostEqual(result[2], 7.5);
            // Should be: result = [2.50, 5.00, 7.50]
        });
        
        runTest("Scalar multiplication assignment operator (*=)", [&]() {
            double arr[] = {1.0, 2.0, 3.0};
            Vector v(arr, 3);
            v *= 2.5;
            return almostEqual(v[0], 2.5) && almostEqual(v[1], 5.0) && almostEqual(v[2], 7.5);
            // Should be: v = [2.50, 5.00, 7.50]
        });
        
        runTest("Dot product operator (*)", [&]() {
            double arr1[] = {1.0, 2.0, 3.0};
            double arr2[] = {4.0, 5.0, 6.0};
            Vector v1(arr1, 3);
            Vector v2(arr2, 3);
            double dot = v1 * v2;
            return almostEqual(dot, 32.0);  // 1*4 + 2*5 + 3*6 = 32
            // Should be: dot product = 32.00
        });
    }
    
    void testUtility() {
        this->printSectionHeader("Utility Function Tests");
        
        runTest("toString()", [&]() {
            double arr[] = {1.5, 2.5, 3.5};
            Vector v(arr, 3);
            string str = v.toString();
            return str.find("1.50") != string::npos && 
                   str.find("2.50") != string::npos && 
                   str.find("3.50") != string::npos;
            // Should be: "[ 1.50, 2.50, 3.50 ]"
        });
        
        runTest("ToMatrix()", [&]() {
            double arr[] = {1.5, 2.5, 3.5};
            Vector v(arr, 3);
            Matrix m = v.ToMatrix();
            return (m.GetNumRows() == 3 && 
                    m.GetNumCols() == 1 &&
                    almostEqual(m(1,1), 1.5) && 
                    almostEqual(m(2,1), 2.5) && 
                    almostEqual(m(3,1), 3.5));
            // Should be: 3x1 Matrix with values 1.50, 2.50, 3.50 in column 1
        });
    }
    
    void testErrorHandling() {
        this->printSectionHeader("Error Handling Tests");
        
        runTest("Invalid size constructor", [&]() {
            bool exceptionCaught = false;
            try {
                Vector v(0);  // Should throw an exception, not assert
            } catch (const std::exception& e) {
                exceptionCaught = true;
            }
            return exceptionCaught;
        });
        
        runTest("Invalid [] indexing", [&]() {
            Vector v(3);
            bool exceptionCaught = false;
            try {
                double x = v[-1];  // Should assert
            } catch (...) {
                exceptionCaught = true;
            }
            return exceptionCaught;
            // Should be: Exception caught (assert failure)
        });
        
        runTest("Invalid () indexing (below 1)", [&]() {
            Vector v(3);
            bool exceptionCaught = false;
            try {
                double x = v(0);  // Should assert, () is 1-based
            } catch (...) {
                exceptionCaught = true;
            }
            return exceptionCaught;
            // Should be: Exception caught (assert failure)
        });
        
        runTest("Invalid () indexing (above size)", [&]() {
            Vector v(3);
            bool exceptionCaught = false;
            try {
                double x = v(4);  // Should assert
            } catch (...) {
                exceptionCaught = true;
            }
            return exceptionCaught;
            // Should be: Exception caught (assert failure)
        });
        
        runTest("Vector size mismatch in binary operations", [&]() {
            Vector v1(3), v2(4);
            bool exceptionCaught = false;
            try {
                Vector sum = v1 + v2;  // Different sizes should assert
            } catch (...) {
                exceptionCaught = true;
            }
            return exceptionCaught;
            // Should be: Exception caught (assert failure)
        });
    }
    
    void runAllTests() {
        printSuiteHeader();

        testConstructors();
        testAccessors();
        testAssignment();
        testUnaryOperators();
        testBinaryOperators();
        testUtility();
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
    VectorTestSuite tests;
    tests.runAllTests();
    return 0;
}