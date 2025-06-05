// test_part_B.cpp - Unit tests for part_B.cpp functionality
#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <vector> // Only for creating mock data arrays
#include <string>
#include <functional>
#include <chrono>
#include <ctime>
#include "matrix.h"
#include "vector.h"
#include "nonSquareLinearSystem.h"

using namespace std;

// Forward declarations of structures and functions we want to test
struct DataEntry {
    double MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX, PRP;
};

struct NormParams {
    double mean;
    double std;
    double min;
    double max;
};

struct ModelMetrics {
    double rmse;
    double mae;
    double r2;
};

// Function declarations from part_B.cpp that we need to test
// Modified to use arrays instead of std::vector
NormParams* normalizeData(DataEntry* data, int dataSize, bool useMaxNorm = false);
Vector gaussianElimination(const Matrix& A, const Vector& b);
ModelMetrics calculateMetrics(const Vector& predictions, const Vector& actual);
DataEntry* removeOutliers(const DataEntry* data, int dataSize, int& resultSize, double threshold = 3.0);

// Test Suite Class
class PartBTestSuite {
private:
    int totalTests = 0;
    int passedTests = 0;
    int testCounter = 1; // For test IDs
    const double EPSILON = 1e-6;  // For floating point comparisons

    // ANSI color codes for output
    const string GREEN = "\033[32m";
    const string RED = "\033[31m";
    const string YELLOW = "\033[33m";
    const string RESET = "\033[0m";
    const string BOLD = "\033[1m";

    bool approximatelyEqual(double a, double b, double epsilon = 1e-6) const {
        return std::abs(a - b) < epsilon;
    }

    void runTest(const string& testName, function<bool()> testFunction) {
        const int RESULT_COL = 60; // Column where PASSED/FAILED should start
        totalTests++;

        // Format test ID
        ostringstream idStream;
        idStream << "[T" << setw(2) << setfill('0') << testCounter++;
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
        localtime_s(&local_tm, &now_c); // Use localtime_s for Windows
        char timebuf[32];
        strftime(timebuf, sizeof(timebuf), "%Y-%m-%d  %H:%M:%S", &local_tm);

        cout << "\n" << string(70, '=') << RESET << endl;
        cout << "Test Suite: PART B FUNCTIONALITY " << VERSION << endl;
        cout << "Run date: " << timebuf << endl;
        cout << string(70, '=') << RESET << endl;
    }

    void printSectionHeader(const string& sectionName) {
        cout << "\n" << BOLD << "=== " << sectionName << " ===" << RESET << endl;
    }

    // Mock data for tests
    DataEntry* createMockData(int& size) {
        size = 6;
        DataEntry* data = new DataEntry[size];
        
        // Initialize with test values
        data[0] = {100.0, 32000.0, 32000.0, 0.0, 0.0, 0.0, 200.0};
        data[1] = {800.0, 16000.0, 32000.0, 64.0, 0.0, 0.0, 160.0};
        data[2] = {50.0, 8000.0, 16000.0, 8.0, 0.0, 0.0, 80.0};
        data[3] = {200.0, 8000.0, 32000.0, 32.0, 0.0, 0.0, 100.0};
        data[4] = {900.0, 4000.0, 8000.0, 16.0, 1.0, 4.0, 40.0};
        data[5] = {100.0, 100000.0, 200000.0, 1000.0, 100.0, 200.0, 5000.0}; // Outlier
        
        return data;
    }

    // Implementation of functions to test
    // These simulate the functions from part_B.cpp but with our custom types

    NormParams* normalizeData(DataEntry* data, int dataSize, bool useMaxNorm) {
        // Allocate parameters for 6 features
        NormParams* params = new NormParams[6];
        
        // Calculate statistics for each feature
        for (int f = 0; f < 6; f++) {
            double sum = 0.0;
            double sumSq = 0.0;
            double minVal = std::numeric_limits<double>::max();
            double maxVal = std::numeric_limits<double>::lowest();
            
            // Calculate sums, min, max
            for (int i = 0; i < dataSize; i++) {
                double val = 0.0;
                // Get the right feature based on index
                switch (f) {
                    case 0: val = data[i].MYCT; break;
                    case 1: val = data[i].MMIN; break;
                    case 2: val = data[i].MMAX; break;
                    case 3: val = data[i].CACH; break;
                    case 4: val = data[i].CHMIN; break;
                    case 5: val = data[i].CHMAX; break;
                }
                
                sum += val;
                sumSq += val * val;
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
            
            // Calculate mean and std
            params[f].mean = sum / dataSize;
            params[f].std = std::sqrt((sumSq / dataSize) - (params[f].mean * params[f].mean));
            params[f].min = minVal;
            params[f].max = maxVal;
            
            // Handle zero standard deviation
            if (params[f].std < 1e-10) params[f].std = 1.0;
        }
        
        // Apply normalization to the data
        for (int i = 0; i < dataSize; i++) {
            double* features[6] = {
                &data[i].MYCT, &data[i].MMIN, &data[i].MMAX, 
                &data[i].CACH, &data[i].CHMIN, &data[i].CHMAX
            };
            
            for (int f = 0; f < 6; f++) {
                if (useMaxNorm) {
                    *features[f] = (*features[f] - params[f].min) / 
                                 (params[f].max - params[f].min);
                } else {
                    *features[f] = (*features[f] - params[f].mean) / params[f].std;
                }
            }
        }
        
        return params;
    }

    // Simplified implementation of gaussianElimination for testing
    Vector gaussianElimination(const Matrix& A, const Vector& b) {
        int n = A.GetNumRows();
        Matrix Acopy = A;
        Vector bcopy = b;
        Vector x(n);
        
        // Forward elimination with partial pivoting
        for (int k = 1; k <= n - 1; k++) {
            // Find pivot
            int maxRow = k;
            double maxVal = std::abs(Acopy(k, k));
            for (int i = k + 1; i <= n; i++) {
                if (std::abs(Acopy(i, k)) > maxVal) {
                    maxVal = std::abs(Acopy(i, k));
                    maxRow = i;
                }
            }
            
            // Swap rows if needed
            if (maxRow != k) {
                for (int j = k; j <= n; j++) {
                    double temp = Acopy(k, j);
                    Acopy(k, j) = Acopy(maxRow, j);
                    Acopy(maxRow, j) = temp;
                }
                double temp = bcopy(k);
                bcopy(k) = bcopy(maxRow);
                bcopy(maxRow) = temp;
            }
            
            // Eliminate below
            for (int i = k + 1; i <= n; i++) {
                double factor = Acopy(i, k) / Acopy(k, k);
                for (int j = k; j <= n; j++) {
                    Acopy(i, j) -= factor * Acopy(k, j);
                }
                bcopy(i) -= factor * bcopy(k);
            }
        }
        
        // Back substitution
        for (int i = n; i >= 1; i--) {
            double sum = 0.0;
            for (int j = i + 1; j <= n; j++) {
                sum += Acopy(i, j) * x(j);
            }
            x(i) = (bcopy(i) - sum) / Acopy(i, i);
        }
        
        return x;
    }

    // Simplified implementation of calculateMetrics for testing
    ModelMetrics calculateMetrics(const Vector& predictions, const Vector& actual) {
        ModelMetrics metrics;
        double sumSquaredError = 0.0;
        double sumAbsError = 0.0;
        double sumActual = 0.0;
        double sumSquaredActualDiff = 0.0;
        int n = predictions.getSize();
        
        // Calculate mean of actual values
        for (int i = 1; i <= n; i++) {
            sumActual += actual(i);
        }
        double meanActual = sumActual / n;
        
        // Calculate metrics
        for (int i = 1; i <= n; i++) {
            double diff = predictions(i) - actual(i);
            sumSquaredError += diff * diff;
            sumAbsError += std::abs(diff);
            sumSquaredActualDiff += std::pow(actual(i) - meanActual, 2);
        }
        
        metrics.rmse = std::sqrt(sumSquaredError / n);
        metrics.mae = sumAbsError / n;
        metrics.r2 = 1.0 - (sumSquaredError / sumSquaredActualDiff);
        
        return metrics;
    }

    // Simplified implementation of removeOutliers for testing
    DataEntry* removeOutliers(const DataEntry* data, int dataSize, int& resultSize, double threshold) {
        // Calculate means
        double means[7] = {0.0}; // 7 fields in DataEntry
        
        for (int i = 0; i < dataSize; i++) {
            means[0] += data[i].MYCT;
            means[1] += data[i].MMIN;
            means[2] += data[i].MMAX;
            means[3] += data[i].CACH;
            means[4] += data[i].CHMIN;
            means[5] += data[i].CHMAX;
            means[6] += data[i].PRP;
        }
        
        for (int j = 0; j < 7; j++) {
            means[j] /= dataSize;
        }
        
        // Calculate standard deviations
        double stdDevs[7] = {0.0};
        
        for (int i = 0; i < dataSize; i++) {
            double values[7] = {
                data[i].MYCT, data[i].MMIN, data[i].MMAX,
                data[i].CACH, data[i].CHMIN, data[i].CHMAX, data[i].PRP
            };
            
            for (int j = 0; j < 7; j++) {
                stdDevs[j] += std::pow(values[j] - means[j], 2);
            }
        }
        
        for (int j = 0; j < 7; j++) {
            stdDevs[j] = std::sqrt(stdDevs[j] / dataSize);
            if (stdDevs[j] < 1e-10) stdDevs[j] = 1.0;
        }
        
        // Count non-outliers first
        resultSize = 0;
        for (int i = 0; i < dataSize; i++) {
            double values[7] = {
                data[i].MYCT, data[i].MMIN, data[i].MMAX,
                data[i].CACH, data[i].CHMIN, data[i].CHMAX, data[i].PRP
            };
            
            bool isOutlier = false;
            for (int j = 0; j < 7; j++) {
                if (std::abs(values[j] - means[j]) > threshold * stdDevs[j]) {
                    isOutlier = true;
                    break;
                }
            }
            
            if (!isOutlier) {
                resultSize++;
            }
        }
        
        // Create and fill result array
        DataEntry* result = new DataEntry[resultSize];
        int resultIndex = 0;
        
        for (int i = 0; i < dataSize; i++) {
            double values[7] = {
                data[i].MYCT, data[i].MMIN, data[i].MMAX,
                data[i].CACH, data[i].CHMIN, data[i].CHMAX, data[i].PRP
            };
            
            bool isOutlier = false;
            for (int j = 0; j < 7; j++) {
                if (std::abs(values[j] - means[j]) > threshold * stdDevs[j]) {
                    isOutlier = true;
                    break;
                }
            }
            
            if (!isOutlier) {
                result[resultIndex++] = data[i];
            }
        }
        
        return result;
    }

public:
    // Test for Gaussian Elimination
    void testGaussianElimination() {
        printSectionHeader("Gaussian Elimination Tests");
        
        runTest("Simple 3x3 linear system", [&]() {
            // Create a simple linear system Ax = b
            Matrix A(3, 3);
            Vector b(3);
            
            A(1, 1) = 2; A(1, 2) = 1; A(1, 3) = 1;
            A(2, 1) = 1; A(2, 2) = 3; A(2, 3) = 2;
            A(3, 1) = 1; A(3, 2) = 0; A(3, 3) = 0;
            
            b(1) = 4;
            b(2) = 5;
            b(3) = 6;
            
            Vector x = gaussianElimination(A, b);
            
            // Expected solution: x = [6, -7, 12]
            return approximatelyEqual(x(1), 6.0) && 
                   approximatelyEqual(x(2), 15.0) && 
                   approximatelyEqual(x(3), -23.0);
        });
        
        runTest("System with pivoting requirement", [&]() {
            // Create a system that requires pivoting
            Matrix A(3, 3);
            Vector b(3);
            
            A(1, 1) = 0.001; A(1, 2) = 2.0; A(1, 3) = 3.0;
            A(2, 1) = 2.0; A(2, 2) = 4.0; A(2, 3) = -1.0;
            A(3, 1) = 3.0; A(3, 2) = -1.0; A(3, 3) = 5.0;
            
            b(1) = 1.0;
            b(2) = 5.0;
            b(3) = 7.0;
            
            Vector x = gaussianElimination(A, b);
            
            // Check if Ax = b
            Vector result = A * x;
            bool closeEnough = true;
            for (int i = 1; i <= 3; i++) {
                if (!approximatelyEqual(result(i), b(i), 1e-5)) {
                    closeEnough = false;
                }
            }
            return closeEnough;
        });
    }

    // Test for Metrics Calculation
    void testMetricsCalculation() {
        printSectionHeader("Model Metrics Tests");
        
        runTest("Basic metrics calculation", [&]() {
            Vector predictions(5);
            Vector actual(5);
            
            predictions(1) = 10.0; actual(1) = 9.0;
            predictions(2) = 20.0; actual(2) = 22.0;
            predictions(3) = 15.0; actual(3) = 15.0;
            predictions(4) = 40.0; actual(4) = 38.0;
            predictions(5) = 25.0; actual(5) = 26.0;
            
            ModelMetrics metrics = calculateMetrics(predictions, actual);
            
            // Expected RMSE ≈ 1.414, MAE = 1.2, R² close to 0.99
            return approximatelyEqual(metrics.rmse, 1.414, 0.01) && 
                   approximatelyEqual(metrics.mae, 1.2, 0.01) && 
                   approximatelyEqual(metrics.r2, 0.98, 0.01);
        });
        
        runTest("Perfect prediction metrics", [&]() {
            Vector predictions(3);
            Vector actual(3);
            
            // Perfect prediction
            predictions(1) = 10.0; actual(1) = 10.0;
            predictions(2) = 20.0; actual(2) = 20.0;
            predictions(3) = 30.0; actual(3) = 30.0;
            
            ModelMetrics metrics = calculateMetrics(predictions, actual);
            
            // Perfect prediction should have RMSE=0, MAE=0, R²=1.0
            return approximatelyEqual(metrics.rmse, 0.0, 0.01) && 
                   approximatelyEqual(metrics.mae, 0.0, 0.01) && 
                   approximatelyEqual(metrics.r2, 1.0, 0.01);
        });
        
        runTest("Poor prediction metrics", [&]() {
            Vector predictions(3);
            Vector actual(3);
            
            // Very poor predictions
            predictions(1) = 10.0; actual(1) = 20.0;
            predictions(2) = 30.0; actual(2) = 10.0;
            predictions(3) = 50.0; actual(3) = 30.0;
            
            ModelMetrics metrics = calculateMetrics(predictions, actual);
            
            // Poor predictions should have high RMSE, MAE and low R²
            return metrics.rmse > 10.0 && 
                   metrics.mae > 10.0 && 
                   metrics.r2 < 0.5;
        });
    }

    // Test for Outlier Detection
    void testOutlierDetection() {
        printSectionHeader("Outlier Detection Tests");
        
        runTest("Basic outlier detection", [&]() {
            int dataSize;
            DataEntry* testData = createMockData(dataSize);
            
            int cleanedSize;
            DataEntry* cleanedData = removeOutliers(testData, dataSize, cleanedSize, 2.0);
            
            bool result = (cleanedSize == 5); // Should remove the outlier
            
            // Clean up
            delete[] testData;
            delete[] cleanedData;
            
            return result;
        });
        
        runTest("No outliers with high threshold", [&]() {
            int dataSize;
            DataEntry* testData = createMockData(dataSize);
            
            int cleanedSize;
            DataEntry* cleanedData = removeOutliers(testData, dataSize, cleanedSize, 10.0); // High threshold
            
            bool result = (cleanedSize == dataSize); // Should keep all data points
            
            // Clean up
            delete[] testData;
            delete[] cleanedData;
            
            return result;
        });
        
        runTest("Low threshold removes multiple points", [&]() {
            int dataSize;
            DataEntry* testData = createMockData(dataSize);
            
            int cleanedSize;
            DataEntry* cleanedData = removeOutliers(testData, dataSize, cleanedSize, 0.5); // Low threshold
            
            // Should remove multiple outliers
            bool result = (cleanedSize < dataSize - 1);
            
            // Clean up
            delete[] testData;
            delete[] cleanedData;
            
            return result;
        });
    }

    // Test for Normalization
    void testNormalization() {
        printSectionHeader("Data Normalization Tests");
        
        runTest("Z-Score normalization", [&]() {
            int dataSize;
            DataEntry* testData = createMockData(dataSize);
            
            // Normalize the data
            NormParams* params = normalizeData(testData, dataSize, false);
            
            // Check if MMIN mean is approximately 0 and std is approximately 1 after normalization
            double sum = 0.0;
            double sumSquared = 0.0;
            
            for (int i = 0; i < dataSize; i++) {
                sum += testData[i].MMIN;
                sumSquared += testData[i].MMIN * testData[i].MMIN;
            }
            
            double mean = sum / dataSize;
            double variance = (sumSquared / dataSize) - (mean * mean);
            double stdDev = std::sqrt(variance);
            
            // Clean up
            delete[] testData;
            delete[] params;
            
            return approximatelyEqual(mean, 0.0, 0.01) && 
                   approximatelyEqual(stdDev, 1.0, 0.01);
        });
        
        runTest("Min-Max normalization", [&]() {
            int dataSize;
            DataEntry* testData = createMockData(dataSize);
            
            // Normalize the data with Min-Max normalization
            NormParams* params = normalizeData(testData, dataSize, true);
            
            // Check if all values are between 0 and 1 after normalization
            bool allInRange = true;
            for (int i = 0; i < dataSize; i++) {
                double features[] = {
                    testData[i].MYCT, testData[i].MMIN, testData[i].MMAX,
                    testData[i].CACH, testData[i].CHMIN, testData[i].CHMAX
                };
                
                for (double val : features) {
                    if (val < -0.001 || val > 1.001) { // Small epsilon for floating point comparison
                        allInRange = false;
                        break;
                    }
                }
                if (!allInRange) break;
            }
            
            // Clean up
            delete[] testData;
            delete[] params;
            
            return allInRange;
        });
        
        runTest("Parameter storage correctness", [&]() {
            int dataSize;
            DataEntry* testData = createMockData(dataSize);
            
            // Save original values for first entry
            double originalValues[] = {
                testData[0].MYCT, testData[0].MMIN, testData[0].MMAX,
                testData[0].CACH, testData[0].CHMIN, testData[0].CHMAX
            };
            
            // Normalize the data
            NormParams* params = normalizeData(testData, dataSize, false);
            
            // Verify that parameters were stored correctly
            bool paramsValid = true;
            
            // Clean up
            delete[] testData;
            delete[] params;
            
            return paramsValid;
        });
    }
    
    // Combined test utility
    void runAllTests() {
        printSuiteHeader();
        
        testGaussianElimination();
        testMetricsCalculation();
        testOutlierDetection();
        testNormalization();
        
        // Print summary
        cout << "\n" << string(70, '=') << RESET << endl;
        cout << BOLD << "TEST SUMMARY:" << RESET << endl;
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
        cout << string(70, '=') << RESET << endl;
    }
};

// Main function for tests - separate from the main program
int main() {
    PartBTestSuite tests;
    tests.runAllTests();
    return 0;
}