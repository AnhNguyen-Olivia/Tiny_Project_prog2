#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cassert>
#include <functional>
#include <iomanip>
#include <algorithm>
#include <random>
#include <limits>
#include "matrix.h"
#include "vector.h"
#include "nonSquareLinearSystem.h"

using namespace std;

// Include the struct definition
struct DataEntry {
    double MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX, PRP;
};

// Structure to hold normalization parameters
struct NormParams {
    double mean;
    double std;
    double min;
    double max;
};

// Structure for model metrics
struct ModelMetrics {
    double rmse;
    double mae;
    double r2;
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

// Function to normalize data
vector<NormParams> test_normalizeData(vector<DataEntry>& data, bool useMaxNorm = false) {
    vector<NormParams> params(6); // 6 features
    vector<double> sums(6, 0.0);
    vector<double> sumSquares(6, 0.0);
    vector<double> mins(6, numeric_limits<double>::max());
    vector<double> maxs(6, numeric_limits<double>::lowest());
    
    // Calculate sums, mins and maxs
    for (const auto& entry : data) {
        vector<double> features = {
            entry.MYCT, entry.MMIN, entry.MMAX, 
            entry.CACH, entry.CHMIN, entry.CHMAX
        };
        
        for (int i = 0; i < 6; i++) {
            sums[i] += features[i];
            sumSquares[i] += features[i] * features[i];
            mins[i] = std::min(mins[i], features[i]);
            maxs[i] = std::max(maxs[i], features[i]);
        }
    }
    
    // Calculate means and standard deviations
    int n = data.size();
    for (int i = 0; i < 6; i++) {
        params[i].mean = sums[i] / n;
        params[i].std = sqrt((sumSquares[i] / n) - (params[i].mean * params[i].mean));
        params[i].min = mins[i];
        params[i].max = maxs[i];
        if (params[i].std == 0) params[i].std = 1; // Prevent division by zero
    }
    
    // Apply normalization
    for (auto& entry : data) {
        if (useMaxNorm) {
            // Max-value normalization (feature / max_value)
            entry.MYCT = entry.MYCT / params[0].max;
            entry.MMIN = entry.MMIN / params[1].max;
            entry.MMAX = entry.MMAX / params[2].max;
            entry.CACH = entry.CACH / params[3].max;
            entry.CHMIN = entry.CHMIN / params[4].max;
            entry.CHMAX = entry.CHMAX / params[5].max;
        } else {
            // Z-score normalization ((feature - mean) / std)
            entry.MYCT = (entry.MYCT - params[0].mean) / params[0].std;
            entry.MMIN = (entry.MMIN - params[1].mean) / params[1].std;
            entry.MMAX = (entry.MMAX - params[2].mean) / params[2].std;
            entry.CACH = (entry.CACH - params[3].mean) / params[3].std;
            entry.CHMIN = (entry.CHMIN - params[4].mean) / params[4].std;
            entry.CHMAX = (entry.CHMAX - params[5].mean) / params[5].std;
        }
    }
    
    return params;
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

// Calculate multiple metrics
ModelMetrics test_calculateMetrics(const Vector& predictions, const Vector& actual) {
    ModelMetrics metrics;
    double sumSquaredError = 0.0;
    double sumAbsError = 0.0;
    double sumActual = 0.0;
    double sumSquaredActualDiff = 0.0;
    int n = predictions.getSize();
    
    // First pass - calculate mean of actual values
    for (int i = 0; i < n; ++i) {
        sumActual += actual[i];
    }
    double meanActual = sumActual / n;
    
    // Second pass - calculate metrics
    for (int i = 0; i < n; ++i) {
        double diff = predictions[i] - actual[i];
        sumSquaredError += diff * diff;
        sumAbsError += abs(diff);
        sumSquaredActualDiff += pow(actual[i] - meanActual, 2);
    }
    
    metrics.rmse = sqrt(sumSquaredError / n);
    metrics.mae = sumAbsError / n;
    metrics.r2 = 1.0 - (sumSquaredError / sumSquaredActualDiff);
    
    return metrics;
}

// Original RMSE function kept for compatibility
double test_calculateRMSE(const Vector& predictions, const Vector& actual) {
    double sum = 0.0;
    for (int i = 0; i < predictions.getSize(); ++i) {
        double diff = predictions[i] - actual[i];
        sum += diff * diff;
    }
    return sqrt(sum / predictions.getSize());
}

// Function to expand features with transformations
Matrix test_expandFeatures(const vector<DataEntry>& data, bool addInteractions = true, 
                          bool addPolynomials = true, bool addLog = true) {
    int baseFeatures = 7; // intercept + 6 original features
    int extraFeatures = (addPolynomials ? 6 : 0) + (addLog ? 6 : 0) + (addInteractions ? 15 : 0);
    int totalFeatures = baseFeatures + extraFeatures;
    
    Matrix X(data.size(), totalFeatures);
    
    // Fill the matrix with data
    for (size_t i = 0; i < data.size(); ++i) {
        const auto& entry = data[i];
        int col = 1;
        
        // Base features
        X(i+1, col++) = 1.0;  // Intercept
        X(i+1, col++) = entry.MYCT;
        X(i+1, col++) = entry.MMIN;
        X(i+1, col++) = entry.MMAX;
        X(i+1, col++) = entry.CACH;
        X(i+1, col++) = entry.CHMIN;
        X(i+1, col++) = entry.CHMAX;
        
        // Polynomial terms
        if (addPolynomials) {
            X(i+1, col++) = entry.MYCT * entry.MYCT;
            X(i+1, col++) = entry.MMIN * entry.MMIN;
            X(i+1, col++) = entry.MMAX * entry.MMAX;
            X(i+1, col++) = entry.CACH * entry.CACH;
            X(i+1, col++) = entry.CHMIN * entry.CHMIN;
            X(i+1, col++) = entry.CHMAX * entry.CHMAX;
        }
        
        // Log transforms
        if (addLog) {
            X(i+1, col++) = log(entry.MYCT + 1.0);
            X(i+1, col++) = log(entry.MMIN + 1.0);
            X(i+1, col++) = log(entry.MMAX + 1.0);
            X(i+1, col++) = log(entry.CACH + 1.0);
            X(i+1, col++) = log(entry.CHMIN + 1.0);
            X(i+1, col++) = log(entry.CHMAX + 1.0);
        }
        
        // Interaction terms
        if (addInteractions) {
            X(i+1, col++) = entry.MYCT * entry.MMIN;
            X(i+1, col++) = entry.MYCT * entry.MMAX;
            X(i+1, col++) = entry.MYCT * entry.CACH;
            X(i+1, col++) = entry.MYCT * entry.CHMIN;
            X(i+1, col++) = entry.MYCT * entry.CHMAX;
            X(i+1, col++) = entry.MMIN * entry.MMAX;
            X(i+1, col++) = entry.MMIN * entry.CACH;
            X(i+1, col++) = entry.MMIN * entry.CHMIN;
            X(i+1, col++) = entry.MMIN * entry.CHMAX;
            X(i+1, col++) = entry.MMAX * entry.CACH;
            X(i+1, col++) = entry.MMAX * entry.CHMIN;
            X(i+1, col++) = entry.MMAX * entry.CHMAX;
            X(i+1, col++) = entry.CACH * entry.CHMIN;
            X(i+1, col++) = entry.CACH * entry.CHMAX;
            X(i+1, col++) = entry.CHMIN * entry.CHMAX;
        }
    }
    
    return X;
}

// Function to remove outliers
vector<DataEntry> test_removeOutliers(const vector<DataEntry>& data, double threshold = 3.0) {
    vector<double> means(7, 0.0), stdDevs(7, 0.0);
    
    // Calculate means
    for (const auto& entry : data) {
        means[0] += entry.MYCT;
        means[1] += entry.MMIN;
        means[2] += entry.MMAX;
        means[3] += entry.CACH;
        means[4] += entry.CHMIN;
        means[5] += entry.CHMAX;
        means[6] += entry.PRP;
    }
    
    for (int i = 0; i < 7; ++i) means[i] /= data.size();
    
    // Calculate standard deviations
    for (const auto& entry : data) {
        const vector<double> values = {
            entry.MYCT, entry.MMIN, entry.MMAX, entry.CACH, 
            entry.CHMIN, entry.CHMAX, entry.PRP
        };
        
        for (int i = 0; i < 7; ++i) {
            stdDevs[i] += pow(values[i] - means[i], 2);
        }
    }
    
    for (int i = 0; i < 7; ++i) {
        stdDevs[i] = sqrt(stdDevs[i] / data.size());
        if (stdDevs[i] < 1e-10) stdDevs[i] = 1.0; // Avoid division by zero
    }
    
    // Filter outliers
    vector<DataEntry> cleanedData;
    
    for (const auto& entry : data) {
        const vector<double> values = {
            entry.MYCT, entry.MMIN, entry.MMAX, entry.CACH, 
            entry.CHMIN, entry.CHMAX, entry.PRP
        };
        
        bool isOutlier = false;
        for (int j = 0; j < 7; ++j) {
            if (abs((values[j] - means[j]) / stdDevs[j]) > threshold) {
                isOutlier = true;
                break;
            }
        }
        
        if (!isOutlier) cleanedData.push_back(entry);
    }
    
    return cleanedData;
}

// Gaussian Elimination with partial pivoting
Vector test_gaussianElimination(const Matrix& A, const Vector& b) {
    int n = A.GetNumRows();
    
    // Create copies we can modify
    Matrix Acopy = A;
    Vector bcopy = b;
    Vector x(n);
    
    // Forward elimination with partial pivoting
    for (int k = 1; k <= n - 1; k++) {
        // Find pivot
        int maxRow = k;
        double maxVal = abs(Acopy(k, k));
        for (int i = k + 1; i <= n; i++) {
            if (abs(Acopy(i, k)) > maxVal) {
                maxVal = abs(Acopy(i, k));
                maxRow = i;
            }
        }
        
        // Check for singularity
        if (maxVal < 1e-10) {
            return x;
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

// Alternative implementation of linear regression using gradient descent
Vector gradientDescentLinearRegression(const Matrix& X, const Vector& y, 
                                      double learningRate = 0.01,
                                      int maxIterations = 100000,
                                      double convergenceThreshold = 1e-14) {
    int numSamples = X.GetNumRows();
    int numFeatures = X.GetNumCols();
    Vector theta(numFeatures);
    
    // Set initial theta values to zeros
    for (int i = 0; i < numFeatures; i++) {
        theta[i] = 0.0;
    }
    
    double prevCost = 1e9;
    
    // Gradient descent iterations
    // Use momentum to improve convergence
    Vector prevGradients(numFeatures);
    double momentum = 0.9;
    
    for (int iter = 0; iter < maxIterations; iter++) {
        // Calculate predictions
        Vector predictions(numSamples);
        for (int i = 0; i < numSamples; i++) {
            predictions[i] = 0;
            for (int j = 0; j < numFeatures; j++) {
                predictions[i] += X(i+1, j+1) * theta[j];
            }
        }
        
        // Calculate gradients with better numerical stability
        Vector gradients(numFeatures);
        for (int j = 0; j < numFeatures; j++) {
            gradients[j] = 0;
            for (int i = 0; i < numSamples; i++) {
                gradients[j] += (predictions[i] - y[i]) * X(i+1, j+1);
            }
            gradients[j] /= numSamples;
        }
        
        // Apply momentum and adaptive learning rate
        double adaptiveLR = learningRate / sqrt(1.0 + iter / 1000.0);
        for (int j = 0; j < numFeatures; j++) {
            theta[j] -= adaptiveLR * (gradients[j] + momentum * prevGradients[j]);
        }
        
        // Save gradients for momentum
        prevGradients = gradients;
        
        // Calculate cost for convergence check
        double cost = 0;
        for (int i = 0; i < numSamples; i++) {
            double diff = predictions[i] - y[i];
            cost += diff * diff;
        }
        cost /= (2 * numSamples);
        
        // Check for convergence
        if (abs(prevCost - cost) < convergenceThreshold) {
            break;
        }
        prevCost = cost;
    }
    
    return theta;
}

class PartBTestSuite {
private:
    int totalTests = 0;
    int passedTests = 0;
    const double EPSILON = 1e-6;  // For floating point comparisons
    std::string currentErrorMsg;  // For storing error messages

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
    
    void setErrorMsg(const string& msg) {
        currentErrorMsg = msg;
    }

    void runTest(const string& testName, function<bool()> testFunction) {
        const int RESULT_COL = 62; // Column where PASS/FAIL/WARN starts
        totalTests++;
        currentErrorMsg = "";

        // Print test name
        cout << "  " << testName;

        // Calculate and print dots
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
                if (!currentErrorMsg.empty()) {
                    cout << "    Error: " << currentErrorMsg << endl;
                }
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

            remove("test_machine.data");
            return correct;
        });

        runTest("Handle empty CSV file", [&]() {
            ofstream testFile("empty.data");
            testFile.close();

            auto data = test_loadData("empty.data");
            bool correct = data.empty();
            
            remove("empty.data");
            return correct;
        });
    }

    void testDataNormalization() {
        printSectionHeader("Data Normalization Tests");
        
        runTest("Z-score normalization", [&]() {
            vector<DataEntry> data(3);
            
            // Setup test data with known mean and std
            data[0].MYCT = 1.0; data[0].MMIN = 10.0;
            data[1].MYCT = 2.0; data[1].MMIN = 20.0;
            data[2].MYCT = 3.0; data[2].MMIN = 30.0;
            
            // Means: MYCT = 2.0, MMIN = 20.0
            // Std: MYCT = 0.816, MMIN = 8.16
            
            auto params = test_normalizeData(data, false); // Z-score
            
            bool meansCorrect = almostEqual(params[0].mean, 2.0, 0.01) && 
                               almostEqual(params[1].mean, 20.0, 0.01);
            
            bool stdsCorrect = almostEqual(params[0].std, 0.8165, 0.01) && 
                              almostEqual(params[1].std, 8.165, 0.01);
            
            bool normalizedCorrect = 
                almostEqual(data[0].MYCT, -1.225, 0.01) && 
                almostEqual(data[0].MMIN, -1.225, 0.01) &&
                almostEqual(data[2].MYCT, 1.225, 0.01) && 
                almostEqual(data[2].MMIN, 1.225, 0.01);
                
            return meansCorrect && stdsCorrect && normalizedCorrect;
        });
        
        runTest("Max-value normalization", [&]() {
            vector<DataEntry> data(3);
            
            // Setup test data
            data[0].MYCT = 1.0; data[0].MMIN = 10.0;
            data[1].MYCT = 2.0; data[1].MMIN = 20.0;
            data[2].MYCT = 3.0; data[2].MMIN = 30.0;
            
            auto params = test_normalizeData(data, true); // Max-value
            
            bool maxCorrect = almostEqual(params[0].max, 3.0) && 
                             almostEqual(params[1].max, 30.0);
            
            bool normalizedCorrect = 
                almostEqual(data[0].MYCT, 1.0/3.0) && 
                almostEqual(data[0].MMIN, 1.0/3.0) &&
                almostEqual(data[2].MYCT, 1.0) && 
                almostEqual(data[2].MMIN, 1.0);
                
            return maxCorrect && normalizedCorrect;
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
            
            test_trainTestSplit(data, train, test, 0.2);
            
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

    void testOutlierRemoval() {
        printSectionHeader("Outlier Removal Tests");
        
        runTest("Remove outliers using Z-score", [&]() {
            vector<DataEntry> data(10);
            
            // Create normal data points
            for (int i = 0; i < 8; i++) {
                data[i].MYCT = 10 + i;
                data[i].MMIN = 100 + i * 10;
                data[i].PRP = 50 + i * 5;
            }
            
            // Create outliers (far from others)
            data[8].MYCT = 100; // Clear outlier
            data[8].MMIN = 120;
            data[8].PRP = 70;
            
            data[9].MYCT = 12; // Not an outlier
            data[9].MMIN = 1000; // Clear outlier
            data[9].PRP = 500; // Clear outlier
            
            auto cleanData = test_removeOutliers(data, 3.0);
            
            // We should have removed the two entries with outliers
            return cleanData.size() == 8;
        });
        
        runTest("Keep all data with high threshold", [&]() {
            vector<DataEntry> data(10);
            
            // Create data with some variance
            for (int i = 0; i < 10; i++) {
                data[i].MYCT = 10 + i * 2;
                data[i].MMIN = 100 + i * 20;
                data[i].PRP = 50 + i * 10;
            }
            
            auto cleanData = test_removeOutliers(data, 10.0); // Very high threshold
            
            return cleanData.size() == 10; // Should keep all data points
        });
    }

    void testMetrics() {
        printSectionHeader("Metrics Calculation Tests");

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
        
        runTest("Calculate multiple metrics", [&]() {
            Vector predictions(4);
            Vector actual(4);
            
            predictions[0] = 1.0; actual[0] = 2.0;
            predictions[1] = 2.0; actual[1] = 2.0;
            predictions[2] = 3.0; actual[2] = 4.0;
            predictions[3] = 4.0; actual[3] = 4.0;
            
            ModelMetrics metrics = test_calculateMetrics(predictions, actual);
            
            // RMSE = sqrt((1^2 + 0^2 + 1^2 + 0^2)/4) = sqrt(2/4) = sqrt(0.5) ≈ 0.7071
            // MAE = (1 + 0 + 1 + 0) / 4 = 0.5
            // Mean actual = (2 + 2 + 4 + 4) / 4 = 3.0
            // Sum of squared actual diffs = (2-3)^2 + (2-3)^2 + (4-3)^2 + (4-3)^2 = 1+1+1+1 = 4
            // R² = 1 - 2/4 = 1 - 0.5 = 0.5
            
            bool rmseCorrect = almostEqual(metrics.rmse, sqrt(0.5));
            bool maeCorrect = almostEqual(metrics.mae, 0.5);
            bool r2Correct = almostEqual(metrics.r2, 0.5);
            
            return rmseCorrect && maeCorrect && r2Correct;
        });
    }

    void testFeatureExpansion() {
        printSectionHeader("Feature Expansion Tests");
        
        runTest("Base features only", [&]() {
            vector<DataEntry> data(2);
            data[0].MYCT = 1; data[0].MMIN = 2; 
            data[0].MMAX = 3; data[0].CACH = 4;
            data[0].CHMIN = 5; data[0].CHMAX = 6;
            
            data[1].MYCT = 10; data[1].MMIN = 20;
            data[1].MMAX = 30; data[1].CACH = 40;
            data[1].CHMIN = 50; data[1].CHMAX = 60;
            
            Matrix X = test_expandFeatures(data, false, false, false);
            
            bool correctSize = X.GetNumRows() == 2 && X.GetNumCols() == 7;
            bool correctValues = 
                almostEqual(X(1, 1), 1.0) && // Intercept
                almostEqual(X(1, 2), 1.0) && // MYCT
                almostEqual(X(1, 7), 6.0) && // CHMAX
                almostEqual(X(2, 2), 10.0) && // MYCT for 2nd row
                almostEqual(X(2, 7), 60.0);  // CHMAX for 2nd row
                
            return correctSize && correctValues;
        });
        
        runTest("Expanded features with polynomials", [&]() {
            vector<DataEntry> data(1);
            data[0].MYCT = 2; data[0].MMIN = 3;
            data[0].MMAX = 0; data[0].CACH = 0;
            data[0].CHMIN = 0; data[0].CHMAX = 0;
            
            Matrix X = test_expandFeatures(data, false, true, false);
            
            bool correctSize = X.GetNumRows() == 1 && X.GetNumCols() == 13;
            bool correctValues = 
                almostEqual(X(1, 2), 2.0) && // MYCT
                almostEqual(X(1, 3), 3.0) && // MMIN
                almostEqual(X(1, 8), 4.0) && // MYCT^2
                almostEqual(X(1, 9), 9.0);  // MMIN^2
                
            return correctSize && correctValues;
        });
        
        runTest("Expanded features with all transformations", [&]() {
            vector<DataEntry> data(1);
            data[0].MYCT = 2; data[0].MMIN = 3;
            data[0].MMAX = 0; data[0].CACH = 0;
            data[0].CHMIN = 0; data[0].CHMAX = 0;
            
            Matrix X = test_expandFeatures(data, true, true, true);
            
            int expectedCols = 7 + 6 + 6 + 15; // base + poly + log + interactions
            bool correctSize = X.GetNumRows() == 1 && X.GetNumCols() == expectedCols;
            bool correctValues = 
                almostEqual(X(1, 8), 4.0) && // MYCT^2
                almostEqual(X(1, 14), log(3.0 + 1.0)) && // log(MMIN+1)
                almostEqual(X(1, 20), 6.0); // MYCT*MMIN interaction
                
            return correctSize && correctValues;
        });
    }

    void testGradientDescent() {
        printSectionHeader("Gradient Descent Linear Regression Tests");

        runTest("Simple linear function with gradient descent", [&]() {
            // Create a simple dataset: y = 2x + 3
            Matrix X(10, 2);  // Added column for intercept
            Vector y(10);
            
            for (int i = 0; i < 10; i++) {
                X(i+1, 1) = 1.0;  // Intercept term
                X(i+1, 2) = i;    // x value
                y[i] = 2 * i + 3;
            }
            
            Vector coefficients = gradientDescentLinearRegression(X, y);
            
            // Check if coefficients are close to [3, 2]
            bool c1Match = almostEqual(coefficients[0], 3.0, 0.2);
            bool c2Match = almostEqual(coefficients[1], 2.0, 0.2);
            
            if (!c1Match || !c2Match) {
                std::ostringstream oss;
                oss << "Expected [3.0, 2.0], got [" 
                    << coefficients[0] << ", " << coefficients[1] << "]";
                setErrorMsg(oss.str());
            }
            
            return c1Match && c2Match;
        });

        runTest("Multiple variable linear function with gradient descent", [&]() {
            // Create a simple dataset: y = 2x₁ + 3x₂ + 1
            Matrix X(20, 3);  // Added column for intercept
            Vector y(20);
            
            for (int i = 0; i < 20; i++) {
                X(i+1, 1) = 1.0;      // Intercept term
                X(i+1, 2) = i / 2.0;  // x₁
                X(i+1, 3) = i / 3.0;  // x₂
                y[i] = 2 * X(i+1, 2) + 3 * X(i+1, 3) + 1;
            }
            
            Vector coefficients = gradientDescentLinearRegression(X, y);
            
            // Check if coefficients are close to [1, 2, 3]
            bool c1Match = almostEqual(coefficients[0], 1.0, 0.3);
            bool c2Match = almostEqual(coefficients[1], 2.0, 0.3);
            bool c3Match = almostEqual(coefficients[2], 3.0, 0.3);
            
            if (!c1Match || !c2Match || !c3Match) {
                std::ostringstream oss;
                oss << "Expected [1.0, 2.0, 3.0], got [" 
                    << coefficients[0] << ", " << coefficients[1] 
                    << ", " << coefficients[2] << "]";
                setErrorMsg(oss.str());
            }
            
            return c1Match && c2Match && c3Match;
        });
    }

    void testGaussianElimination() {
        printSectionHeader("Gaussian Elimination Tests");
        
        runTest("Solve simple system with Gaussian elimination", [&]() {
            // Create a simple system: 2x + y = 5, x + 3y = 10
            // Solution: x = 1, y = 3
            Matrix A(2, 2);
            Vector b(2);
            
            A(1, 1) = 2.0; A(1, 2) = 1.0;
            A(2, 1) = 1.0; A(2, 2) = 3.0;
            
            b(1) = 5.0;
            b(2) = 10.0;
            
            Vector solution = test_gaussianElimination(A, b);
            
            bool correctSolution = almostEqual(solution(1), 1.0) && 
                                  almostEqual(solution(2), 3.0);
            
            if (!correctSolution) {
                std::ostringstream oss;
                oss << "Expected [1.0, 3.0], got [" 
                    << solution(1) << ", " << solution(2) << "]";
                setErrorMsg(oss.str());
            }
            
            return correctSolution;
        });
        
        runTest("Solve larger system with Gaussian elimination", [&]() {
            // Create a 3x3 system
            Matrix A(3, 3);
            Vector b(3);
            
            A(1, 1) = 3.0; A(1, 2) = 2.0; A(1, 3) = -1.0;
            A(2, 1) = 2.0; A(2, 2) = -2.0; A(2, 3) = 4.0;
            A(3, 1) = -1.0; A(3, 2) = 0.5; A(3, 3) = -1.0;
            
            b(1) = 1.0;
            b(2) = -2.0;
            b(3) = 0.0;
            
            Vector solution = test_gaussianElimination(A, b);
            
            // Expected solution: x = 1, y = -2, z = -2
            bool correctSolution = 
                almostEqual(solution(1), 1.0) && 
                almostEqual(solution(2), -2.0) &&
                almostEqual(solution(3), -2.0);
            
            if (!correctSolution) {
                std::ostringstream oss;
                oss << "Expected [1.0, -2.0, -2.0], got [" 
                    << solution(1) << ", " << solution(2) << ", " << solution(3) << "]";
                setErrorMsg(oss.str());
            }
            
            return correctSolution;
        });
    }

    void testAlgorithmComparison() {
        printSectionHeader("Algorithm Comparison Tests");

        runTest("Compare gradient descent with pseudo-inverse", [&]() {
            // Create synthetic data with a bit of noise to avoid perfect collinearity
            ofstream testFile("synthetic_compare.data");
            for (int i = 0; i < 50; i++) {
                double myct = i / 10.0;
                double mmin = i / 5.0;
                // Add small random values to other features
                double mmax = (rand() % 100) * 0.01;
                double cach = (rand() % 100) * 0.01;
                double chmin = (rand() % 100) * 0.01;
                double chmax = (rand() % 100) * 0.01;
                double noise = (rand() % 100) * 0.001;
                double prp = 2 * myct + 3 * mmin + 0.1 * mmax + noise;
                testFile << "X,Y," << myct << "," << mmin << "," 
                        << mmax << "," << cach << "," << chmin << "," 
                        << chmax << "," << prp << ",0,0\n";
            }
            testFile.close();
            
            auto data = test_loadData("synthetic_compare.data");
            vector<DataEntry> trainData, testData;
            test_trainTestSplit(data, trainData, testData, 0.2);
            
            // Build matrices for both methods, adding intercept for gradient descent
            Matrix A(trainData.size(), 6);
            Matrix A_gd(trainData.size(), 7); // One extra column for intercept
            Vector b(trainData.size());
            
            for (size_t i = 0; i < trainData.size(); ++i) {
                const auto& entry = trainData[i];
                A(i+1, 1) = entry.MYCT;
                A(i+1, 2) = entry.MMIN;
                A(i+1, 3) = entry.MMAX;
                A(i+1, 4) = entry.CACH;
                A(i+1, 5) = entry.CHMIN;
                A(i+1, 6) = entry.CHMAX;
                
                A_gd(i+1, 1) = 1.0;           // Intercept
                A_gd(i+1, 2) = entry.MYCT;
                A_gd(i+1, 3) = entry.MMIN;
                A_gd(i+1, 4) = entry.MMAX;
                A_gd(i+1, 5) = entry.CACH;
                A_gd(i+1, 6) = entry.CHMIN;
                A_gd(i+1, 7) = entry.CHMAX;
                
                b[i] = entry.PRP;  // Using 0-based indexing for vector
            }
            
            // Solve with pseudo-inverse (original method)
            NonSquareLinearSystem solver(A, b);
            Vector coefficients1 = solver.SolveWithPseudoInverse();
            
            // Solve with gradient descent (alternative method)
            Vector coefficients2 = gradientDescentLinearRegression(A_gd, b);
            
            // Now compare predictions from both methods
            Vector predictions1(testData.size());
            Vector predictions2(testData.size());
            Vector actual(testData.size());
            
            for (size_t i = 0; i < testData.size(); ++i) {
                const auto& entry = testData[i];
                actual[i] = entry.PRP;
                
                // Predictions with pseudo-inverse coefficients (1-based indexing)
                predictions1[i] = coefficients1(1) * entry.MYCT + 
                                coefficients1(2) * entry.MMIN + 
                                coefficients1(3) * entry.MMAX + 
                                coefficients1(4) * entry.CACH + 
                                coefficients1(5) * entry.CHMIN + 
                                coefficients1(6) * entry.CHMAX;
                
                // Predictions with gradient descent coefficients (0-based indexing)
                predictions2[i] = coefficients2[0] +             // Intercept
                                coefficients2[1] * entry.MYCT + 
                                coefficients2[2] * entry.MMIN + 
                                coefficients2[3] * entry.MMAX + 
                                coefficients2[4] * entry.CACH + 
                                coefficients2[5] * entry.CHMIN + 
                                coefficients2[6] * entry.CHMAX;
            }
            
            double rmse1 = test_calculateRMSE(predictions1, actual);
            double rmse2 = test_calculateRMSE(predictions2, actual);
            
            remove("synthetic_compare.data"); // Clean up
            
            // Both methods should produce reasonable RMSEs
            bool pass = rmse1 < 0.5 && rmse2 < 0.5;
            
            if (!pass) {
                std::ostringstream oss;
                oss << "RMSE values too high: Pseudo-inverse RMSE = " << rmse1
                    << ", Gradient descent RMSE = " << rmse2;
                setErrorMsg(oss.str());
            }
            
            return pass;
        });
    }

    void testEndToEnd() {
        printSectionHeader("End-to-End Linear Regression Tests");

        runTest("Simple linear regression workflow with pseudo-inverse", [&]() {
            // Create a simple dataset with perfect linear relationship
            // Y = 2*MYCT + 3*MMIN + 0*others
            ofstream testFile("synthetic.data");
            for (int i = 0; i < 50; i++) {
                double myct = i / 10.0;
                double mmin = i / 5.0;
                // Add small random values to avoid singularity
                double mmax = (rand() % 100) * 0.001;
                double cach = (rand() % 100) * 0.001;
                double chmin = (rand() % 100) * 0.001;
                double chmax = (rand() % 100) * 0.001;
                double noise = (rand() % 100) * 0.0001;
                double prp = 2 * myct + 3 * mmin + noise;
                testFile << "X,Y," << myct << "," << mmin << "," 
                        << mmax << "," << cach << "," << chmin << "," 
                        << chmax << "," << prp << ",0,0\n";
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
                b[i] = entry.PRP;  // Changed to 0-based indexing
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
                
                // Use 1-based indexing for coefficients to match how they're produced
                double prediction = coefficients(1) * entry.MYCT +  // Column 1
                                  coefficients(2) * entry.MMIN +    // Column 2
                                  coefficients(3) * entry.MMAX +    // Column 3
                                  coefficients(4) * entry.CACH +    // Column 4
                                  coefficients(5) * entry.CHMIN +   // Column 5
                                  coefficients(6) * entry.CHMAX;    // Column 6
                testPredictions[i] = prediction;
            }
            
            double rmse = test_calculateRMSE(testPredictions, testActual);
            
            remove("synthetic.data"); // Clean up
            
            // For this perfect linear relationship, RMSE should be very small
            // and coefficients should be close to [2, 3, 0, 0, 0, 0]
            bool rmseCheck = rmse < 0.001;
            bool coeffCheck = almostEqual(coefficients(1), 2.0, 0.1) &&
                            almostEqual(coefficients(2), 3.0, 0.1) &&
                            almostEqual(coefficients(3), 0.0, 0.1) &&
                            almostEqual(coefficients(4), 0.0, 0.1) &&
                            almostEqual(coefficients(5), 0.0, 0.1) &&
                            almostEqual(coefficients(6), 0.0, 0.1);
            
            if (!rmseCheck || !coeffCheck) {
                std::ostringstream oss;
                oss << "RMSE = " << rmse << " (expected < 0.001), Coefficients = ["
                    << coefficients(1) << ", " << coefficients(2) << ", "
                    << coefficients(3) << ", " << coefficients(4) << ", "
                    << coefficients(5) << ", " << coefficients(6) << "]";
                setErrorMsg(oss.str());
            }
            
            return rmseCheck && coeffCheck;
        });
        
        runTest("Simple linear regression workflow with gradient descent", [&]() {
            // Create a simple dataset with perfect linear relationship
            // Y = 2*MYCT + 3*MMIN + 0*others
            ofstream testFile("synthetic_gd.data");
            for (int i = 0; i < 50; i++) {
                double myct = i / 10.0;
                double mmin = i / 5.0;
                double prp = 2 * myct + 3 * mmin;
                testFile << "X,Y," << myct << "," << mmin << ",0,0,0,0," << prp << ",0,0\n";
            }
            testFile.close();
            
            // Run the linear regression
            auto data = test_loadData("synthetic_gd.data");
            vector<DataEntry> trainData, testData;
            test_trainTestSplit(data, trainData, testData, 0.2);
            
            // Build matrix A and vector b - add intercept column
            Matrix A(trainData.size(), 7); // 6 features + intercept
            Vector b(trainData.size());
            
            for (size_t i = 0; i < trainData.size(); ++i) {
                const auto& entry = trainData[i];
                A(i+1, 1) = 1.0;           // Intercept
                A(i+1, 2) = entry.MYCT;
                A(i+1, 3) = entry.MMIN;
                A(i+1, 4) = entry.MMAX;
                A(i+1, 5) = entry.CACH;
                A(i+1, 6) = entry.CHMIN;
                A(i+1, 7) = entry.CHMAX;
                b[i] = entry.PRP;  // Use 0-based indexing
            }
            
            // Solve using gradient descent
            Vector coefficients = gradientDescentLinearRegression(A, b);
            
            // Test predictions
            Vector testPredictions(testData.size());
            Vector testActual(testData.size());
            
            for (size_t i = 0; i < testData.size(); ++i) {
                const auto& entry = testData[i];
                testActual[i] = entry.PRP;
                
                // Use 0-based indexing for gradient descent coefficients
                double prediction = coefficients[0] +             // Intercept
                                  coefficients[1] * entry.MYCT +
                                  coefficients[2] * entry.MMIN +
                                  coefficients[3] * entry.MMAX +
                                  coefficients[4] * entry.CACH +
                                  coefficients[5] * entry.CHMIN +
                                  coefficients[6] * entry.CHMAX;
                testPredictions[i] = prediction;
            }
            
            double rmse = test_calculateRMSE(testPredictions, testActual);
            
            remove("synthetic_gd.data"); // Clean up
            
            // For this simple linear relationship, gradient descent should converge to a decent solution
            bool pass = rmse < 0.5;
            
            if (!pass) {
                std::ostringstream oss;
                oss << "RMSE = " << rmse << " (expected < 0.5), Coefficients = ["
                    << coefficients[0] << ", " << coefficients[1] << ", "
                    << coefficients[2] << ", " << coefficients[3] << ", "
                    << coefficients[4] << ", " << coefficients[5] << ", "
                    << coefficients[6] << "]";
                setErrorMsg(oss.str());
            }
            
            return pass;
        });
    }

    void runAllTests() {
        cout << "\n" << string(70, '=') << endl;
        cout << "           PART B (LINEAR REGRESSION) TEST SUITE" << endl;
        cout << string(70, '=') << endl;

        testDataLoading();
        testDataNormalization();
        testTrainTestSplit();
        testOutlierRemoval();
        testMetrics();
        testFeatureExpansion();
        testGaussianElimination();
        testGradientDescent();
        testAlgorithmComparison();
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