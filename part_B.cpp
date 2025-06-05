#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>  // For output formatting
#include <chrono>   // For timing operations
#include "matrix.h"
#include "vector.h"
#include "nonSquareLinearSystem.h"

// ANSI color codes for output
const std::string GREEN = "\033[32m";
const std::string RED = "\033[31m";
const std::string YELLOW = "\033[33m";
const std::string BLUE = "\033[34m";
const std::string MAGENTA = "\033[35m";
const std::string CYAN = "\033[36m";
const std::string BOLD = "\033[1m";
const std::string RESET = "\033[0m";

// Struct to hold a data instance
struct DataEntry {
    double MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX, PRP;
};

// Progress bar function with enhanced visual
void showProgressBar(int progress, int total) {
    const int barWidth = 50;
    float percentage = static_cast<float>(progress) / total;
    int pos = static_cast<int>(barWidth * percentage);
    
    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << CYAN << "=" << RESET;
        else if (i == pos) std::cout << BLUE << ">" << RESET;
        else std::cout << " ";
    }
    std::cout << "] " << BOLD << int(percentage * 100.0) << "%" << RESET << "\r";
    std::cout.flush();
    
    if (progress == total) {
        std::cout << std::endl;
    }
}

// Print section header with consistent styling
void printHeader(const std::string& title) {
    const int WIDTH = 70;
    std::cout << "\n" << BLUE << std::string(WIDTH, '=') << RESET << std::endl;
    std::cout << BOLD << "  " << title << RESET << std::endl;
    std::cout << BLUE << std::string(WIDTH, '=') << RESET << std::endl;
}

// Function to load data from CSV with progress indication
std::vector<DataEntry> loadData(const std::string& filename) {
    std::vector<DataEntry> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << RED << "ERROR: Could not open file " << filename << RESET << std::endl;
        return data;
    }
    
    printHeader("DATA LOADING");
    std::cout << "Source: " << CYAN << filename << RESET << std::endl;
    
    // Count lines for progress bar
    int totalLines = 0;
    std::string line;
    std::ifstream countFile(filename);
    while (std::getline(countFile, line)) totalLines++;
    countFile.close();
    
    int currentLine = 0;
    while (std::getline(file, line)) {
        // Show progress
        if (totalLines > 100 && currentLine % (totalLines/100) == 0) {
            showProgressBar(currentLine, totalLines);
        }
        
        std::stringstream ss(line);
        std::string token;
        DataEntry entry;

        // Skip first two columns (non-predictive)
        for (int i = 0; i < 2; ++i) std::getline(ss, token, ',');

        // Read predictive features
        std::getline(ss, token, ','); entry.MYCT = std::stod(token);
        std::getline(ss, token, ','); entry.MMIN = std::stod(token);
        std::getline(ss, token, ','); entry.MMAX = std::stod(token);
        std::getline(ss, token, ','); entry.CACH = std::stod(token);
        std::getline(ss, token, ','); entry.CHMIN = std::stod(token);
        std::getline(ss, token, ','); entry.CHMAX = std::stod(token);

        // Read target (PRP)
        std::getline(ss, token, ','); entry.PRP = std::stod(token);

        data.push_back(entry);
        currentLine++;
    }

    if (totalLines > 100) showProgressBar(totalLines, totalLines);
    std::cout << GREEN << "[SUCCESS] " << RESET << "Loaded " << BOLD << data.size() 
              << RESET << " data entries" << std::endl;
    return data;
}

// Structure to hold normalization parameters
struct NormParams {
    double mean;
    double std;
    double min;
    double max;
};

// Function that implements Gaussian Elimination with partial pivoting
Vector gaussianElimination(const Matrix& A, const Vector& b) {
    int n = A.GetNumRows();
    
    // Create copies we can modify
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
        
        // Check for singularity
        if (maxVal < 1e-10) {
            std::cerr << RED << "Warning: Matrix may be singular or ill-conditioned" << RESET << std::endl;
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

// Function to normalize a dataset with enhanced feedback
std::vector<NormParams> normalizeData(std::vector<DataEntry>& data, bool useMaxNorm = false) {
    printHeader("DATA NORMALIZATION");
    
    std::vector<NormParams> params(6); // 6 features
    std::vector<double> sums(6, 0.0);
    std::vector<double> sumSquares(6, 0.0);
    std::vector<double> mins(6, std::numeric_limits<double>::max());
    std::vector<double> maxs(6, std::numeric_limits<double>::lowest());
    
    // Calculate sums, mins and maxs
    for (const auto& entry : data) {
        std::vector<double> features = {
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
    std::cout << "Using " << (useMaxNorm ? "max-value" : "z-score") << " normalization..." << std::endl;
    
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
    
    // Print normalization info
    std::cout << BOLD << "Feature Statistics:" << RESET << std::endl;
    std::cout << CYAN << std::setw(10) << "Feature" << std::setw(12) << "Mean" << std::setw(12) << "StdDev" 
              << std::setw(12) << "Min" << std::setw(12) << "Max" << RESET << std::endl;
    std::cout << std::string(58, '-') << std::endl;
    
    std::vector<std::string> featureNames = {"MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"};
    for (int i = 0; i < 6; i++) {
        std::cout << YELLOW << std::setw(10) << featureNames[i] << RESET
                  << std::setw(12) << std::fixed << std::setprecision(4) << params[i].mean 
                  << std::setw(12) << params[i].std
                  << std::setw(12) << params[i].min
                  << std::setw(12) << params[i].max << std::endl;
    }
    std::cout << GREEN << "[SUCCESS] " << RESET << "Normalization complete" << std::endl;
    
    return params;
}

// Split data into training and test sets with random shuffling
void trainTestSplit(std::vector<DataEntry>& data, 
                   std::vector<DataEntry>& train,
                   std::vector<DataEntry>& test,
                   double testSize = 0.2) {
    printHeader("DATASET SPLITTING");
    
    // Shuffle the data
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
    
    int n = data.size();
    int testSamples = static_cast<int>(n * testSize);
    
    train = std::vector<DataEntry>(data.begin(), data.end() - testSamples);
    test = std::vector<DataEntry>(data.end() - testSamples, data.end());
    
    std::cout << GREEN << "[SUCCESS] " << RESET << "Data split into:" << std::endl;
    std::cout << "  - " << CYAN << "Training: " << RESET << BOLD << train.size() << RESET << " samples" << std::endl;
    std::cout << "  - " << YELLOW << "Testing:  " << RESET << BOLD << test.size() << RESET << " samples" << std::endl;
    std::cout << "  - " << MAGENTA << "Ratio:    " << RESET << std::fixed << std::setprecision(1) 
              << (100 - testSize * 100) << "% / " << (testSize * 100) << "%" << std::endl;
}

// Calculate RMSE and other metrics
struct ModelMetrics {
    double rmse;
    double mae;
    double r2;
};

ModelMetrics calculateMetrics(const Vector& predictions, const Vector& actual) {
    ModelMetrics metrics;
    double sumSquaredError = 0.0;
    double sumAbsError = 0.0;
    double sumActual = 0.0;
    double sumSquaredActual = 0.0;
    int n = predictions.getSize();
    
    for (int i = 0; i < n; ++i) {
        double diff = predictions[i] - actual[i];
        sumSquaredError += diff * diff;
        sumAbsError += std::abs(diff);
        sumActual += actual[i];
        sumSquaredActual += actual[i] * actual[i];
    }
    
    // Calculate metrics
    metrics.rmse = std::sqrt(sumSquaredError / n);
    metrics.mae = sumAbsError / n;
    
    // Calculate R-squared
    double meanActual = sumActual / n;
    double totalSumSquares = 0.0;
    for (int i = 0; i < n; ++i) {
        totalSumSquares += std::pow(actual[i] - meanActual, 2);
    }
    
    metrics.r2 = 1.0 - (sumSquaredError / totalSumSquares);
    
    return metrics;
}

void printModelSummary(const Vector& coefficients, 
                      const std::vector<NormParams>& normParams,
                      const ModelMetrics& metrics,
                      bool useMaxNorm = false,
                      const std::string& methodName = "Pseudo-Inverse") {
    printHeader("MODEL SUMMARY: " + methodName);
    
    // Print metrics
    std::cout << BOLD << "Model Performance Metrics:" << RESET << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << std::setw(20) << "Metric" << std::setw(20) << "Value" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << CYAN << std::setw(20) << "RMSE:" << RESET << std::setw(20) << std::fixed << std::setprecision(4) << metrics.rmse << std::endl;
    std::cout << CYAN << std::setw(20) << "MAE:" << RESET << std::setw(20) << std::fixed << std::setprecision(4) << metrics.mae << std::endl;
    std::cout << CYAN << std::setw(20) << "R^2:" << RESET << std::setw(20) << std::fixed << std::setprecision(4) << metrics.r2 << std::endl;
    std::cout << std::string(40, '-') << std::endl << std::endl;
    
    // Print coefficients
    std::cout << BOLD << "Model Coefficients (Normalized Scale):" << RESET << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << std::setw(20) << "Parameter" << std::setw(20) << "Value" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << MAGENTA << std::setw(20) << "Intercept:" << RESET << std::setw(20) << std::fixed << std::setprecision(6) << coefficients(1) << std::endl;
    
    std::vector<std::string> featureNames = {"MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"};
    for (int i = 0; i < 6; i++) {
        std::cout << MAGENTA << std::setw(20) << featureNames[i] + ":" << RESET << std::setw(20) << std::fixed << std::setprecision(6) << coefficients(i+2) << std::endl;
    }
    std::cout << std::string(40, '-') << std::endl << std::endl;
    
    // Print original scale coefficients
    std::cout << BOLD << "Model Coefficients (Original Scale):" << RESET << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << std::setw(20) << "Parameter" << std::setw(20) << "Value" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    double originalIntercept = coefficients(1);
    for (int i = 0; i < 6; i++) {
        if (useMaxNorm) {
            originalIntercept -= coefficients(i+2) * 0; // For max normalization, no mean adjustment
        } else {
            originalIntercept -= coefficients(i+2) * normParams[i].mean / normParams[i].std;
        }
    }
    std::cout << YELLOW << std::setw(20) << "Intercept:" << RESET << std::setw(20) << std::fixed << std::setprecision(6) << originalIntercept << std::endl;
    
    for (int i = 0; i < 6; i++) {
        double originalCoef;
        if (useMaxNorm) {
            // FIXED: Changed 'params' to 'normParams'
            originalCoef = coefficients(i+2) / normParams[i].max;
        } else {
            originalCoef = coefficients(i+2) / normParams[i].std;
        }
        std::cout << YELLOW << std::setw(20) << featureNames[i] + ":" << RESET << std::setw(20) << std::fixed << std::setprecision(6) << originalCoef << std::endl;
    }
    std::cout << std::string(40, '-') << std::endl << std::endl;
    
    // Print the linear regression equation
    std::cout << BOLD << "Linear Regression Equation:" << RESET << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << BLUE << "PRP = " << std::fixed << std::setprecision(4) << originalIntercept;
    for (int i = 0; i < 6; i++) {
        double coef;
        if (useMaxNorm) {
            // FIXED: Changed 'params' to 'normParams'
            coef = coefficients(i+2) / normParams[i].max;
        } else {
            coef = coefficients(i+2) / normParams[i].std;
        }
        if (coef >= 0) std::cout << " + ";
        else std::cout << " - ";
        std::cout << std::abs(coef) << " * " << featureNames[i];
    }
    std::cout << RESET << std::endl;
    std::cout << std::string(70, '-') << std::endl;
}

// Display application version and runtime info
void displayVersionInfo() {
    const std::string VERSION = "v1.1.0";
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    
    std::cout << BLUE << "\n" << std::string(70, '*') << RESET << std::endl;
    std::cout << BOLD << "COMPUTER HARDWARE PERFORMANCE PREDICTION " << VERSION << RESET << std::endl;
    std::cout << "Execution date: " << std::ctime(&now_c);
    std::cout << BLUE << std::string(70, '*') << RESET << std::endl;
}

// Compare different solution methods
void compareMethodsTable(const std::vector<ModelMetrics>& allMetrics, const std::vector<std::string>& methodNames) {
    printHeader("SOLUTION METHODS COMPARISON");
    
    std::cout << BOLD << std::setw(25) << "Method" << std::setw(15) << "RMSE" 
              << std::setw(15) << "MAE" << std::setw(15) << "R^2" << RESET << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (size_t i = 0; i < methodNames.size(); i++) {
        std::string color = (i == 0) ? CYAN : (i == 1) ? YELLOW : MAGENTA;
        std::cout << color << std::setw(25) << methodNames[i] << RESET
                  << std::setw(15) << std::fixed << std::setprecision(4) << allMetrics[i].rmse
                  << std::setw(15) << std::fixed << std::setprecision(4) << allMetrics[i].mae
                  << std::setw(15) << std::fixed << std::setprecision(4) << allMetrics[i].r2 << std::endl;
    }
    std::cout << std::string(70, '-') << std::endl;
    
    // Find best method
    size_t bestIndex = 0;
    for (size_t i = 1; i < allMetrics.size(); i++) {
        if (allMetrics[i].rmse < allMetrics[bestIndex].rmse) {
            bestIndex = i;
        }
    }
    
    std::cout << GREEN << "Best method: " << BOLD << methodNames[bestIndex] 
              << RESET << GREEN << " with RMSE of " << std::fixed << std::setprecision(4) 
              << allMetrics[bestIndex].rmse << RESET << std::endl;
}

int main() {
    displayVersionInfo();
    auto startTime = std::chrono::high_resolution_clock::now();

    // Load data
    auto data = loadData("machine.data");
    if (data.empty()) {
        std::cerr << RED << "[ERROR] " << RESET << "No data loaded. Exiting." << std::endl;
        return 1;
    }
    
    // Use z-score normalization (mean=0, std=1)
    auto normParams = normalizeData(data, false);
    
    // Split into train/test
    std::vector<DataEntry> trainData, testData;
    trainTestSplit(data, trainData, testData);

    // Build matrix A and vector b for training
    printHeader("MODEL BUILDING");
    std::cout << "Creating linear system from training data..." << std::endl;
    
    Matrix A(trainData.size(), 7); // 7 columns (intercept + 6 features)
    Vector b(trainData.size());
    
    for (size_t i = 0; i < trainData.size(); ++i) {
        const auto& entry = trainData[i];
        A(i+1, 1) = 1.0;  // Intercept term
        A(i+1, 2) = entry.MYCT;
        A(i+1, 3) = entry.MMIN;
        A(i+1, 4) = entry.MMAX;
        A(i+1, 5) = entry.CACH;
        A(i+1, 6) = entry.CHMIN;
        A(i+1, 7) = entry.CHMAX;
        b(i+1) = entry.PRP;  // Use 1-based indexing with () operator
    }

    // Vectors to store results from multiple methods
    std::vector<Vector> allCoefficients;
    std::vector<ModelMetrics> allMetrics;
    std::vector<std::string> methodNames;

    // METHOD 1: Solve using Pseudo-Inverse
    std::cout << "Solving linear system with pseudo-inverse method..." << std::endl;
    NonSquareLinearSystem solver(A, b);
    Vector coefficients_pseudoinv = solver.SolveWithPseudoInverse();
    std::cout << GREEN << "[SUCCESS] " << RESET << "Pseudo-inverse solution calculated" << std::endl;
    allCoefficients.push_back(coefficients_pseudoinv);
    methodNames.push_back("Pseudo-Inverse");

    // METHOD 2: Solve using Normal Equations + Gaussian elimination
    std::cout << "Solving linear system with normal equations..." << std::endl;
    Matrix AtA = A.Transpose() * A;
    Vector Atb = A.Transpose() * b;
    Vector coefficients_gaussian = gaussianElimination(AtA, Atb);
    std::cout << GREEN << "[SUCCESS] " << RESET << "Gaussian elimination solution calculated" << std::endl;
    allCoefficients.push_back(coefficients_gaussian);
    methodNames.push_back("Normal Equations");

    // METHOD 3: Solve using Tikhonov regularization (ridge regression)
    std::cout << "Solving linear system with Tikhonov regularization (lambda=0.1)..." << std::endl;
    Vector coefficients_tikhonov = solver.SolveWithTikhonov(0.1);
    std::cout << GREEN << "[SUCCESS] " << RESET << "Tikhonov regularization solution calculated" << std::endl;
    allCoefficients.push_back(coefficients_tikhonov);
    methodNames.push_back("Tikhonov Regularization");

    // Prepare test data and evaluate all methods
    printHeader("MODEL EVALUATION");
    std::cout << "Evaluating all methods on test set..." << std::endl;
    
    // Create test matrices/vectors
    Vector testActual(testData.size());
    std::vector<Vector> testPredictions;
    
    // Evaluate each method
    for (size_t method = 0; method < allCoefficients.size(); method++) {
        Vector predictions(testData.size());
        
        for (size_t i = 0; i < testData.size(); ++i) {
            const auto& entry = testData[i];
            if (method == 0) testActual[i] = entry.PRP; // Only need to do this once
            
            // Calculate prediction with current method's coefficients
            double prediction = allCoefficients[method](1) + // Intercept
                            allCoefficients[method](2) * entry.MYCT +
                            allCoefficients[method](3) * entry.MMIN +
                            allCoefficients[method](4) * entry.MMAX +
                            allCoefficients[method](5) * entry.CACH +
                            allCoefficients[method](6) * entry.CHMIN +
                            allCoefficients[method](7) * entry.CHMAX;
            predictions[i] = prediction;
        }
        
        testPredictions.push_back(predictions);
        allMetrics.push_back(calculateMetrics(predictions, testActual));
    }

    std::cout << GREEN << "[SUCCESS] " << RESET << "Model evaluation complete" << std::endl;
    
    // Display comparison of all methods
    compareMethodsTable(allMetrics, methodNames);
    
    // Display detailed results for best method (lowest RMSE)
    size_t bestIndex = 0;
    for (size_t i = 1; i < allMetrics.size(); i++) {
        if (allMetrics[i].rmse < allMetrics[bestIndex].rmse) {
            bestIndex = i;
        }
    }
    
    // Display model summary for the best method
    printModelSummary(allCoefficients[bestIndex], normParams, allMetrics[bestIndex], false, methodNames[bestIndex] + " (Best)");
    
    // Display execution time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    printHeader("EXECUTION SUMMARY");
    std::cout << "Total execution time: " << BOLD << duration / 1000.0 << " seconds" << RESET << std::endl;
    std::cout << "Program completed " << GREEN << "successfully" << RESET << std::endl;
    std::cout << BLUE << std::string(70, '=') << RESET << std::endl;

    return 0;
}