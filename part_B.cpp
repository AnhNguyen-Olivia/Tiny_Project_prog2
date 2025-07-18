#include <iostream>
#include <fstream>
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
    double MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX, PRP;           // Hardware features and target (PRP)
};

// Progress bar function with enhanced visual
void showProgressBar(int progress, int total) {
    const int barWidth = 50;                                    // Width of the progress bar in characters
    float percentage = static_cast<float>(progress) / total;    // Calculate progress percentage
    int pos = static_cast<int>(barWidth * percentage);          // Calculate how many characters to fill

    std::cout << "["; // Start of progress bar
    for (int i = 0; i < barWidth; ++i) { // Loop through the bar width
        if (i < pos) std::cout << CYAN << "=" << RESET; // Completed part of the bar in cyan
        else if (i == pos) std::cout << BLUE << ">" << RESET; // Current position marker in blue
        else std::cout << " "; // Remaining part is empty
    }
    std::cout << "] " << BOLD << int(percentage * 100.0) << "%" << RESET << "\r"; // Print percentage
    std::cout.flush(); // Ensure output is displayed immediately

    if (progress == total) { // If complete, print a newline to move to next line
        std::cout << std::endl;
    }
}

// Print section header with consistent styling
void printHeader(const std::string& title) {
    const int WIDTH = 70; // Total width for header decoration
    std::cout << "\n" << BLUE << std::string(WIDTH, '=') << RESET << std::endl; // Top border
    std::cout << BOLD << "  " << title << RESET << std::endl; // Title in bold
    std::cout << BLUE << std::string(WIDTH, '=') << RESET << std::endl; // Bottom border
}

// Function to load data from CSV with progress indication
std::vector<DataEntry> loadData(const std::string& filename) {
    std::vector<DataEntry> data; // Vector to store loaded data
    std::ifstream file(filename); // Open file for reading

    if (!file.is_open()) { // Check if file couldn't be opened
        std::cerr << RED << "ERROR: Could not open file " << filename << RESET << std::endl;
        return data; // Return empty data vector
    }

    printHeader("DATA LOADING"); // Display loading header
    std::cout << "Source: " << CYAN << filename << RESET << std::endl; // Show source filename

    // Reserve initial capacity to avoid reallocations
    data.reserve(1000); // Preallocate space for performance

    std::string line; // To store each line of the file
    int totalLines = 0; // Count how many lines read

    while (std::getline(file, line)) { // Read file line by line
        totalLines++; // Increment line count

        // Show progress every 100 lines
        if (totalLines % 100 == 0) {
            showProgressBar(totalLines, std::max(totalLines, 1000)); // Display progress bar
        }

        std::stringstream ss(line); // Create stream from line
        std::string token; // Temporary token for each CSV field
        DataEntry entry; // New entry to store parsed data

        // Skip first two columns (non-predictive) - use getline directly
        std::getline(ss, token, ','); // Skip first column
        std::getline(ss, token, ','); // Skip second column

        // Read predictive features and target from the remaining columns
        std::getline(ss, token, ','); entry.MYCT = std::stod(token); // Convert and assign MYCT
        std::getline(ss, token, ','); entry.MMIN = std::stod(token); // Convert and assign MMIN
        std::getline(ss, token, ','); entry.MMAX = std::stod(token); // Convert and assign MMAX
        std::getline(ss, token, ','); entry.CACH = std::stod(token); // Convert and assign CACH
        std::getline(ss, token, ','); entry.CHMIN = std::stod(token); // Convert and assign CHMIN
        std::getline(ss, token, ','); entry.CHMAX = std::stod(token); // Convert and assign CHMAX
        std::getline(ss, token, ','); entry.PRP = std::stod(token);   // Convert and assign PRP

        data.push_back(entry); // Add entry to dataset
    }

    showProgressBar(totalLines, totalLines); // Final call to show complete progress bar
    std::cout << GREEN << "[SUCCESS] " << RESET << "Loaded " << BOLD << data.size() 
              << RESET << " data entries" << std::endl; // Report success and number of entries
    return data; // Return loaded data
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

// Split data into training and test sets with fixed seed for reproducibility
void trainTestSplit(std::vector<DataEntry>& data, 
                   std::vector<DataEntry>& train,
                   std::vector<DataEntry>& test,
                   double testSize = 0.2) {
    printHeader("DATASET SPLITTING");
    
    // Shuffle the data with a fixed seed
    std::mt19937 g(42); // Fixed seed for reproducibility
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
    double sumSquaredActualDiff = 0.0;
    int n = predictions.getSize();
    int validPredictions = 0;
    double meanActual = 0.0;
    
    // First pass - calculate mean of actual values
    for (int i = 1; i <= n; ++i) {
        if (!std::isnan(actual(i))) {
            sumActual += actual(i);
            validPredictions++;
        }
    }
    
    if (validPredictions == 0) {
        metrics.rmse = metrics.mae = metrics.r2 = std::numeric_limits<double>::quiet_NaN();
        return metrics;
    }
    
    meanActual = sumActual / validPredictions;
    
    // Second pass - calculate metrics in one go
    for (int i = 1; i <= n; ++i) {
        if (std::isnan(predictions(i)) || std::isnan(actual(i))) continue;
        
        double diff = predictions(i) - actual(i);
        sumSquaredError += diff * diff;
        sumAbsError += std::abs(diff);
        sumSquaredActualDiff += std::pow(actual(i) - meanActual, 2);
    }
    
    metrics.rmse = std::sqrt(sumSquaredError / validPredictions);
    metrics.mae = sumAbsError / validPredictions;
    metrics.r2 = (sumSquaredActualDiff < 1e-10) ? 0.0 : 1.0 - (sumSquaredError / sumSquaredActualDiff);
    
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
    
    // Handle NaN values in metrics
    if (std::isnan(metrics.rmse)) {
        std::cout << CYAN << std::setw(20) << "RMSE:" << RESET << std::setw(20) << "Error" << std::endl;
    } else {
        std::cout << CYAN << std::setw(20) << "RMSE:" << RESET << std::setw(20) << std::fixed << std::setprecision(4) << metrics.rmse << std::endl;
    }
    
    if (std::isnan(metrics.mae)) {
        std::cout << CYAN << std::setw(20) << "MAE:" << RESET << std::setw(20) << "Error" << std::endl;
    } else {
        std::cout << CYAN << std::setw(20) << "MAE:" << RESET << std::setw(20) << std::fixed << std::setprecision(4) << metrics.mae << std::endl;
    }
    
    if (std::isnan(metrics.r2)) {
        std::cout << CYAN << std::setw(20) << "R^2:" << RESET << std::setw(20) << "Error" << std::endl;
    } else {
        std::cout << CYAN << std::setw(20) << "R^2:" << RESET << std::setw(20) << std::fixed << std::setprecision(4) << metrics.r2 << std::endl;
    }
    
    std::cout << std::string(40, '-') << std::endl << std::endl;
    
    // Check if coefficients contain NaN values
    bool hasNaN = false;
    for (int i = 1; i <= coefficients.getSize(); i++) {
        if (std::isnan(coefficients(i))) {
            hasNaN = true;
            break;
        }
    }
    
    if (hasNaN) {
        std::cout << RED << "[ERROR] " << RESET << "Model coefficients contain NaN values. Unable to display model details." << std::endl;
        return;
    }
    
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

// Function to expand features with transformations and interactions
Matrix expandFeatures(const std::vector<DataEntry>& data, bool addInteractions = true, 
                     bool addPolynomials = true, bool addLog = true) {
    int baseFeatures = 7; // intercept + 6 original features
    int extraFeatures = (addPolynomials ? 6 : 0) + (addLog ? 6 : 0) + (addInteractions ? 15 : 0);
    int totalFeatures = baseFeatures + extraFeatures;
    
    Matrix X(data.size(), totalFeatures);
    
    // Fill the matrix with data - use more efficient looping
    for (size_t i = 0; i < data.size(); ++i) {
        const auto& entry = data[i];
        int col = 1;
        
        // Base features (always included)
        X(i+1, col++) = 1.0;  // Intercept
        X(i+1, col++) = entry.MYCT;
        X(i+1, col++) = entry.MMIN;
        X(i+1, col++) = entry.MMAX;
        X(i+1, col++) = entry.CACH;
        X(i+1, col++) = entry.CHMIN;
        X(i+1, col++) = entry.CHMAX;
        
        // Add polynomial terms (squared features)
        if (addPolynomials) {
            X(i+1, col++) = entry.MYCT * entry.MYCT;
            X(i+1, col++) = entry.MMIN * entry.MMIN;
            X(i+1, col++) = entry.MMAX * entry.MMAX;
            X(i+1, col++) = entry.CACH * entry.CACH;
            X(i+1, col++) = entry.CHMIN * entry.CHMIN;
            X(i+1, col++) = entry.CHMAX * entry.CHMAX;
        }
        
        // Add log transforms - compute logs once
        if (addLog) {
            double logMYCT = std::log(entry.MYCT + 1.0);
            double logMMIN = std::log(entry.MMIN + 1.0);
            double logMMAX = std::log(entry.MMAX + 1.0);
            double logCACH = std::log(entry.CACH + 1.0);
            double logCHMIN = std::log(entry.CHMIN + 1.0);
            double logCHMAX = std::log(entry.CHMAX + 1.0);
            
            X(i+1, col++) = logMYCT;
            X(i+1, col++) = logMMIN;
            X(i+1, col++) = logMMAX;
            X(i+1, col++) = logCACH;
            X(i+1, col++) = logCHMIN;
            X(i+1, col++) = logCHMAX;
        }
        
        // Add interaction terms more efficiently
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

// Function to get feature names for the expanded feature matrix
std::vector<std::string> getExpandedFeatureNames(bool addInteractions = true, 
                                               bool addPolynomials = true,
                                               bool addLog = true) {
    std::vector<std::string> featureNames = {"Intercept", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"};
    const std::vector<std::string> baseFeatures = {"MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"};
    
    // Pre-allocate memory to avoid reallocations
    featureNames.reserve(7 + (addPolynomials ? 6 : 0) + (addLog ? 6 : 0) + (addInteractions ? 15 : 0));
    
    if (addPolynomials)
        for (const auto& name : baseFeatures) featureNames.push_back(name + "²");
    
    if (addLog) 
        for (const auto& name : baseFeatures) featureNames.push_back("log(" + name + ")");
    
    if (addInteractions)
        for (size_t i = 0; i < baseFeatures.size(); ++i)
            for (size_t j = i + 1; j < baseFeatures.size(); ++j)
                featureNames.push_back(baseFeatures[i] + "*" + baseFeatures[j]);
    
    return featureNames;
}

// Function to detect and remove outliers using Z-score method
std::vector<DataEntry> removeOutliers(const std::vector<DataEntry>& data, double threshold = 3.0) {
    printHeader("OUTLIER DETECTION");
    
    const size_t dataSize = data.size();
    std::vector<double> means(7, 0.0), stdDevs(7, 0.0);
    
    // Calculate means in one pass
    for (const auto& entry : data) {
        means[0] += entry.MYCT;
        means[1] += entry.MMIN;
        means[2] += entry.MMAX;
        means[3] += entry.CACH;
        means[4] += entry.CHMIN;
        means[5] += entry.CHMAX;
        means[6] += entry.PRP;
    }
    
    for (int i = 0; i < 7; ++i) means[i] /= dataSize;
    
    // Calculate standard deviations in one pass
    for (const auto& entry : data) {
        const std::vector<double> values = {
            entry.MYCT, entry.MMIN, entry.MMAX, entry.CACH, 
            entry.CHMIN, entry.CHMAX, entry.PRP
        };
        
        for (int i = 0; i < 7; ++i) {
            stdDevs[i] += std::pow(values[i] - means[i], 2);
        }
    }
    
    for (int i = 0; i < 7; ++i) {
        stdDevs[i] = std::sqrt(stdDevs[i] / dataSize);
        if (stdDevs[i] < 1e-10) stdDevs[i] = 1.0; // Avoid division by zero
    }
    
    // Pre-allocate result vector with estimated size
    std::vector<DataEntry> cleanedData;
    cleanedData.reserve(dataSize);
    
    // Identify and filter outliers in one pass
    int outlierCount = 0;
    for (const auto& entry : data) {
        const std::vector<double> values = {
            entry.MYCT, entry.MMIN, entry.MMAX, entry.CACH, 
            entry.CHMIN, entry.CHMAX, entry.PRP
        };
        
        bool isOutlier = false;
        for (int j = 0; j < 7; ++j) {
            if (std::abs((values[j] - means[j]) / stdDevs[j]) > threshold) {
                isOutlier = true;
                outlierCount++;
                break;
            }
        }
        
        if (!isOutlier) cleanedData.push_back(entry);
    }
    
    std::cout << GREEN << "[SUCCESS] " << RESET << "Identified " << BOLD << outlierCount 
              << RESET << " outliers (" << std::fixed << std::setprecision(1) 
              << (100.0 * outlierCount / dataSize) << "% of data)" << std::endl;
    std::cout << "Retained " << BOLD << cleanedData.size() << RESET << " data points" << std::endl;
    
    return cleanedData;
}

// Function to find optimal lambda for Tikhonov regularization using k-fold cross-validation
double findOptimalLambda(const Matrix& A, const Vector& b, int kFolds = 5) {
    printHeader("REGULARIZATION PARAMETER TUNING");
    
    const std::vector<double> lambdaValues = {0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0};
    std::vector<double> avgRmseValues(lambdaValues.size(), 0.0);
    
    const int n = A.GetNumRows();
    const int foldSize = n / kFolds;
    
    // Create indices once
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i + 1;
    
    std::mt19937 g(42); // Fixed seed for reproducibility
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Pre-allocate datasets to avoid repeated allocations
    Matrix trainA(n - foldSize, A.GetNumCols()); 
    Vector trainB(n - foldSize);
    Matrix validA(foldSize, A.GetNumCols());
    Vector validB(foldSize);
    NonSquareLinearSystem solver(trainA, trainB);
    
    for (size_t l = 0; l < lambdaValues.size(); ++l) {
        double lambda = lambdaValues[l];
        double totalRmse = 0.0;
        
        std::cout << "Testing lambda = " << lambda << " ";
        
        // K-fold cross validation
        for (int fold = 0; fold < kFolds; ++fold) {
            int validStartIdx = fold * foldSize + 1;
            int validEndIdx = (fold == kFolds - 1) ? n : (fold + 1) * foldSize;
            
            // Fill train/valid matrices
            int trainRow = 1, validRow = 1;
            for (int i = 1; i <= n; ++i) {
                int idx = indices[i-1];
                if (i >= validStartIdx && i <= validEndIdx) {
                    // Add to validation set
                    for (int j = 1; j <= A.GetNumCols(); ++j) {
                        validA(validRow, j) = A(idx, j);
                    }
                    validB(validRow) = b(idx);
                    validRow++;
                } else {
                    // Add to training set
                    for (int j = 1; j <= A.GetNumCols(); ++j) {
                        trainA(trainRow, j) = A(idx, j);
                    }
                    trainB(trainRow) = b(idx);
                    trainRow++;
                }
            }
            
            // Update solver with new training data (avoiding recreation)
            //solver.UpdateSystem(trainA, trainB);
            //Vector coef = solver.SolveWithTikhonov(lambda);
            
            // Instead of updating an existing solver
            // Create a new solver instance each time
            NonSquareLinearSystem foldSolver(trainA, trainB);
            Vector coef = foldSolver.SolveWithTikhonov(lambda);
            
            // Compute RMSE directly
            double rmse = 0.0;
            for (int i = 1; i <= validB.getSize(); ++i) {
                double prediction = 0;
                for (int j = 1; j <= A.GetNumCols(); ++j) {
                    prediction += coef(j) * validA(i, j);
                }
                rmse += std::pow(prediction - validB(i), 2);
            }
            rmse = std::sqrt(rmse / validB.getSize());
            totalRmse += rmse;
            
            std::cout << ".";
            std::cout.flush();
        }
        
        avgRmseValues[l] = totalRmse / kFolds;
        std::cout << " Avg RMSE: " << std::fixed << std::setprecision(4) << avgRmseValues[l] << std::endl;
    }
    
    // Find best lambda
    size_t bestIdx = std::min_element(avgRmseValues.begin(), avgRmseValues.end()) - avgRmseValues.begin();
    double bestLambda = lambdaValues[bestIdx];
    
    std::cout << GREEN << "[SUCCESS] " << RESET << "Optimal lambda: " << BOLD << bestLambda 
              << RESET << " (CV RMSE: " << avgRmseValues[bestIdx] << ")" << std::endl;
    
    return bestLambda;
}

int main() {
    displayVersionInfo();
    auto startTime = std::chrono::high_resolution_clock::now();

    // Load and prepare data more efficiently
    auto data = loadData("machine.data");
    if (data.empty()) {
        std::cerr << RED << "[ERROR] " << RESET << "No data loaded. Exiting." << std::endl;
        return 1;
    }
    
    data = removeOutliers(data, 4.0); 
    auto normParams = normalizeData(data, false);
    
    // Split into train/test
    std::vector<DataEntry> trainData, testData;
    trainTestSplit(data, trainData, testData);

    printHeader("MODEL BUILDING");
    
    // Create feature matrices more efficiently
    const size_t trainSize = trainData.size();
    const size_t testSize = testData.size();
    Matrix trainX(trainSize, 7); 
    Vector trainY(trainSize);
    Matrix testX(testSize, 7);
    Vector testY(testSize);
    
    // Fill train and test matrices in one loop each
    for (size_t i = 0; i < trainSize; ++i) {
        trainX(i+1, 1) = 1.0;  // Intercept
        trainX(i+1, 2) = trainData[i].MYCT;
        trainX(i+1, 3) = trainData[i].MMIN;
        trainX(i+1, 4) = trainData[i].MMAX;
        trainX(i+1, 5) = trainData[i].CACH;
        trainX(i+1, 6) = trainData[i].CHMIN;
        trainX(i+1, 7) = trainData[i].CHMAX;
        trainY(i+1) = trainData[i].PRP;
    }
    
    for (size_t i = 0; i < testSize; ++i) {
        testX(i+1, 1) = 1.0;  
        testX(i+1, 2) = testData[i].MYCT;
        testX(i+1, 3) = testData[i].MMIN;
        testX(i+1, 4) = testData[i].MMAX;
        testX(i+1, 5) = testData[i].CACH;
        testX(i+1, 6) = testData[i].CHMIN;
        testX(i+1, 7) = testData[i].CHMAX;
        testY(i+1) = testData[i].PRP;
    }

    // Pre-allocate vectors for results
    std::vector<Vector> allCoefficients;
    std::vector<ModelMetrics> allMetrics;
    std::vector<std::string> methodNames = {"Pseudo-Inverse", "Normal Equations", "Tikhonov Regularization"};
    allCoefficients.reserve(3);
    allMetrics.reserve(3);
    
    // METHOD 1: Solve using Pseudo-Inverse
    NonSquareLinearSystem solver(trainX, trainY);
    allCoefficients.push_back(solver.SolveWithPseudoInverse());
    
    // METHOD 2: Solve using Normal Equations + Gaussian elimination
    Matrix AtA = trainX.Transpose() * trainX;
    Vector Atb = trainX.Transpose() * trainY;
    allCoefficients.push_back(gaussianElimination(AtA, Atb));
    
    // METHOD 3: Solve using Tikhonov regularization
    allCoefficients.push_back(solver.SolveWithTikhonov(0.1));

    // Evaluate all methods efficiently
    printHeader("MODEL EVALUATION");
    
    for (size_t method = 0; method < allCoefficients.size(); method++) {
        Vector predictions(testSize);
        const Vector& coef = allCoefficients[method];
        
        // Make predictions more efficiently
        for (size_t i = 1; i <= testSize; ++i) {
            double pred = 0.0;
            for (int j = 1; j <= 7; ++j) {
                pred += coef(j) * testX(i, j);
            }
            predictions(i) = pred;
        }
        
        allMetrics.push_back(calculateMetrics(predictions, testY));
    }
    
    // Find best method with std::min_element
    size_t bestIndex = std::min_element(
        allMetrics.begin(), allMetrics.end(),
        [](const ModelMetrics& a, const ModelMetrics& b) { return a.rmse < b.rmse; }
    ) - allMetrics.begin();
    
    // Display results
    compareMethodsTable(allMetrics, methodNames);
    printModelSummary(allCoefficients[bestIndex], normParams, allMetrics[bestIndex], 
                     false, methodNames[bestIndex] + " (Best)");
    
    // Calculate execution time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    printHeader("EXECUTION SUMMARY");
    std::cout << "Total execution time: " << BOLD << duration / 1000.0 << " seconds" << RESET << std::endl;
    std::cout << "Program completed " << GREEN << "successfully" << RESET << std::endl;
    std::cout << BLUE << std::string(70, '=') << RESET << std::endl;

    return 0;
}