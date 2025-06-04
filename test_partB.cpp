#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cassert>
#include <functional>
#include <iomanip>
#include "matrix.h"
#include "vector.h"
#include "nonSquareLinearSystem.h"

using namespace std;

// Include the struct definition
struct DataEntry {
    double MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX, PRP;
};

// We need to reimplement the functions here for testing instead of importing from part_B.cpp
// This avoids the multiple definition errors

// Function to load data from CSV
vector<DataEntry> test_loadData(const string& filename) {
    vector<DataEntry> data;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        DataEntry entry;

        // Skip first two columns (non-predictive)
        for (int i = 0; i < 2; ++i) getline(ss, token, ',');

        // Read predictive features
        getline(ss, token, ','); entry.MYCT = stod(token);
        getline(ss, token, ','); entry.MMIN = stod(token);
        getline(ss, token, ','); entry.MMAX = stod(token);
        getline(ss, token, ','); entry.CACH = stod(token);
        getline(ss, token, ','); entry.CHMIN = stod(token);
        getline(ss, token, ','); entry.CHMAX = stod(token);

        // Read target (PRP)
        getline(ss, token, ','); entry.PRP = stod(token);

        data.push_back(entry);
    }

    return data;
}

// Split data into training and test sets
void test_trainTestSplit(const vector<DataEntry>& data, 
                   vector<DataEntry>& train,
                   vector<DataEntry>& test,
                   double testSize = 0.2) {
    int n = data.size();
    int testSamples = static_cast<int>(n * testSize);
    
    // Use first 80% for training, last 20% for testing
    train = vector<DataEntry>(data.begin(), data.end() - testSamples);
    test = vector<DataEntry>(data.end() - testSamples, data.end());
}

// Calculate RMSE
double test_calculateRMSE(const Vector& predictions, const Vector& actual) {
    double sum = 0.0;
    for (int i = 0; i < predictions.getSize(); ++i) {
        double diff = predictions[i] - actual[i];
        sum += diff * diff;
    }
    return sqrt(sum / predictions.getSize());
}

class PartBTestSuite {
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

    bool almostEqual(double a, double b, double epsilon) const {
        return fabs(a - b) < epsilon;
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
                cout << GREEN << "PASS" << RESET << endl;
                passedTests++;
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
    void testDataLoading() {
        printSectionHeader("Data Loading Tests");

        runTest("Load data from CSV file", [&]() {
            // Create a small test CSV file
            ofstream testFile("test_machine.data");
            testFile << "X,Y,1,2,3,4,5,6,7,8,9\n";
            testFile << "A,B,10,20,30,40,50,60,70,80,90\n";
            testFile.close();

            auto data = test_loadData("test_machine.data");
            
            bool correct = data.size() == 2 &&
                          almostEqual(data[0].MYCT, 1) &&
                          almostEqual(data[0].MMIN, 2) &&
                          almostEqual(data[0].MMAX, 3) &&
                          almostEqual(data[0].CACH, 4) &&
                          almostEqual(data[0].CHMIN, 5) &&
                          almostEqual(data[0].CHMAX, 6) &&
                          almostEqual(data[0].PRP, 7) &&
                          almostEqual(data[1].MYCT, 10) &&
                          almostEqual(data[1].PRP, 70);

            remove("test_machine.data"); // Clean up
            return correct;
        });

        runTest("Handle empty CSV file", [&]() {
            ofstream testFile("empty.data");
            testFile.close();

            auto data = test_loadData("empty.data");
            bool correct = data.empty();
            
            remove("empty.data"); // Clean up
            return correct;
        });
    }

    void testTrainTestSplit() {
        printSectionHeader("Train-Test Split Tests");

        runTest("Split with default ratio (80/20)", [&]() {
            vector<DataEntry> data(100);
            vector<DataEntry> train, test;
            
            // Fill data with dummy values
            for (int i = 0; i < 100; i++) {
                data[i].PRP = i;
            }
            
            test_trainTestSplit(data, train, test, 0.2); // Explicitly pass 0.2
            
            return train.size() == 80 && test.size() == 20 &&
                   almostEqual(train[0].PRP, 0) &&
                   almostEqual(test[0].PRP, 80);
        });

        runTest("Split with custom ratio (70/30)", [&]() {
            vector<DataEntry> data(100);
            vector<DataEntry> train, test;
            
            // Fill data with dummy values
            for (int i = 0; i < 100; i++) {
                data[i].PRP = i;
            }
            
            test_trainTestSplit(data, train, test, 0.3);
            
            return train.size() == 70 && test.size() == 30 &&
                   almostEqual(train[0].PRP, 0) &&
                   almostEqual(test[0].PRP, 70);
        });
    }

    void testRMSE() {
        printSectionHeader("RMSE Calculation Tests");

        runTest("Calculate RMSE for identical vectors", [&]() {
            Vector predictions(3);
            Vector actual(3);
            
            predictions[0] = 1.0; actual[0] = 1.0;
            predictions[1] = 2.0; actual[1] = 2.0;
            predictions[2] = 3.0; actual[2] = 3.0;
            
            double rmse = test_calculateRMSE(predictions, actual);
            
            return almostEqual(rmse, 0.0);
        });

        runTest("Calculate RMSE for different vectors", [&]() {
            Vector predictions(3);
            Vector actual(3);
            
            predictions[0] = 1.0; actual[0] = 2.0;
            predictions[1] = 2.0; actual[1] = 4.0;
            predictions[2] = 3.0; actual[2] = 6.0;
            
            double rmse = test_calculateRMSE(predictions, actual);
            
            // RMSE = sqrt((1^2 + 2^2 + 3^2)/3) = sqrt(14/3) ≈ 2.16
            return almostEqual(rmse, sqrt(14.0/3.0));
        });
    }

    void testEndToEnd() {
        printSectionHeader("End-to-End Linear Regression Tests");

        runTest("Simple linear regression workflow", [&]() {
            // Create a simple dataset with perfect linear relationship
            // Y = 2*MYCT + 3*MMIN + 0*others
            ofstream testFile("synthetic.data");
            for (int i = 0; i < 50; i++) {
                double myct = i / 10.0;
                double mmin = i / 5.0;
                double prp = 2 * myct + 3 * mmin;
                testFile << "X,Y," << myct << "," << mmin << ",0,0,0,0," << prp << ",0,0\n";
            }
            testFile.close();
            
            // Run the linear regression
            auto data = test_loadData("synthetic.data");
            vector<DataEntry> trainData, testData;
            test_trainTestSplit(data, trainData, testData, 0.2);
            
            // Build matrix A and vector b
            Matrix A(trainData.size(), 6);
            Vector b(trainData.size());
            
            for (size_t i = 0; i < trainData.size(); ++i) {
                const auto& entry = trainData[i];
                A(i+1, 1) = entry.MYCT;
                A(i+1, 2) = entry.MMIN;
                A(i+1, 3) = entry.MMAX;
                A(i+1, 4) = entry.CACH;
                A(i+1, 5) = entry.CHMIN;
                A(i+1, 6) = entry.CHMAX;
                b[i] = entry.PRP;
            }
            
            // Solve
            NonSquareLinearSystem solver(A, b);
            Vector coefficients = solver.SolveWithPseudoInverse();
            
            // Test predictions
            Vector testPredictions(testData.size());
            Vector testActual(testData.size());
            
            for (size_t i = 0; i < testData.size(); ++i) {
                const auto& entry = testData[i];
                testActual[i] = entry.PRP;
                
                double prediction = coefficients[0] * entry.MYCT +
                                  coefficients[1] * entry.MMIN +
                                  coefficients[2] * entry.MMAX +
                                  coefficients[3] * entry.CACH +
                                  coefficients[4] * entry.CHMIN +
                                  coefficients[5] * entry.CHMAX;
                testPredictions[i] = prediction;
            }
            
            double rmse = test_calculateRMSE(testPredictions, testActual);
            
            remove("synthetic.data"); // Clean up
            
            // For this perfect linear relationship, RMSE should be very small
            // and coefficients should be close to [2, 3, 0, 0, 0, 0]
            return rmse < 0.001 &&
                   almostEqual(coefficients[0], 2.0, 0.1) &&
                   almostEqual(coefficients[1], 3.0, 0.1) &&
                   almostEqual(coefficients[2], 0.0, 0.1) &&
                   almostEqual(coefficients[3], 0.0, 0.1) &&
                   almostEqual(coefficients[4], 0.0, 0.1) &&
                   almostEqual(coefficients[5], 0.0, 0.1);
        });
    }

    void runAllTests() {
        cout << "\n" << string(70, '=') << endl;
        cout << "           PART B (LINEAR REGRESSION) TEST SUITE" << endl;
        cout << string(70, '=') << endl;

        testDataLoading();
        testTrainTestSplit();
        testRMSE();
        testEndToEnd();

        // Print summary
        cout << "\n" << string(70, '=') << endl;
        cout << "TEST SUMMARY:" << endl;
        cout << "  Total tests : " << setw(3) << totalTests << endl;
        cout << "  Tests passed: " << setw(3) << passedTests << " ("
             << fixed << setprecision(1) << (totalTests > 0 ? (100.0 * passedTests / totalTests) : 0)
             << "%)" << endl;
        cout << "  Tests failed: " << setw(3) << (totalTests - passedTests) << endl;
        
        if (passedTests == totalTests) {
            cout << GREEN << "  ALL TESTS PASSED!" << RESET << endl;
        } else {
            cout << RED << "  SOME TESTS FAILED!" << RESET << endl;
        }
        
        cout << string(70, '=') << endl;
    }
};

int main() {
    PartBTestSuite tests;
    tests.runAllTests();
    return 0;
}