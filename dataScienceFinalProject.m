%% Breast Cancer Classification using SVM and Decision Tree
% Dataset: Breast Cancer Wisconsin (Diagnostic)
% File: wdbc.csv (32 columns, no header)
% Col1 = ID, Col2 = Diagnosis (M/B), Col3-32 = 30 numeric features

clc; clear; close all;

%% 1. Load and Prepare Dataset
dataFile = "wdbc.csv";   % make sure this file is in your current folder

% Read without headers
Traw = readtable(dataFile, "ReadVariableNames", false);

% Assign official column names
varNames = { ...
    'ID', ...
    'Diagnosis', ...
    'radius_mean', ...
    'texture_mean', ...
    'perimeter_mean', ...
    'area_mean', ...
    'smoothness_mean', ...
    'compactness_mean', ...
    'concavity_mean', ...
    'concave_points_mean', ...
    'symmetry_mean', ...
    'fractal_dimension_mean', ...
    'radius_se', ...
    'texture_se', ...
    'perimeter_se', ...
    'area_se', ...
    'smoothness_se', ...
    'compactness_se', ...
    'concavity_se', ...
    'concave_points_se', ...
    'symmetry_se', ...
    'fractal_dimension_se', ...
    'radius_worst', ...
    'texture_worst', ...
    'perimeter_worst', ...
    'area_worst', ...
    'smoothness_worst', ...
    'compactness_worst', ...
    'concavity_worst', ...
    'concave_points_worst', ...
    'symmetry_worst', ...
    'fractal_dimension_worst' ...
};

Traw.Properties.VariableNames = varNames;


% Diagnosis: 'M'/'B'
Y = categorical(Traw.Diagnosis);

% Features: 30 numeric columns (3..32)
X = Traw{:, 3:end};

classes = categories(Y);
disp("Classes:");
disp(classes);

fprintf("Total samples: %d\n", size(X,1));
disp("Class distribution:");
disp(table(classes, countcats(Y), 'VariableNames', {'Class','Count'}));

%% 2. Train/Test Split (Stratified 80/20)
rng(42);  % reproducibility

cvHoldout = cvpartition(Y, "Holdout", 0.2);
Xtrain = X(training(cvHoldout), :);
Ytrain = Y(training(cvHoldout), :);
Xtest  = X(test(cvHoldout), :);
Ytest  = Y(test(cvHoldout), :);

fprintf("\nTrain size: %d\n", size(Xtrain,1));
fprintf("Test size : %d\n", size(Xtest,1));

%% 3. Standardize Features for SVM
mu = mean(Xtrain, 1);
sigma = std(Xtrain, 0, 1);
sigma(sigma == 0) = 1;

XtrainStd = (Xtrain - mu) ./ sigma;
XtestStd  = (Xtest  - mu) ./ sigma;

%% 4. SVM (RBF) with Simple 10-fold CV Hyperparameter Search

C_values = [0.1, 1, 10];
gamma_values = [0.01, 0.001];

kfold = 10;
cv = cvpartition(Ytrain, "KFold", kfold);

bestAccSVM = -inf;
bestC = NaN;
bestGamma = NaN;
bestKernelScale = NaN;

fprintf("\n=== SVM Hyperparameter Search (10-fold CV) ===\n");

for Ci = 1:numel(C_values)
    for gi = 1:numel(gamma_values)
        C = C_values(Ci);
        gamma = gamma_values(gi);
        % gamma = 1/(2*KernelScale^2) -> KernelScale = 1/sqrt(2*gamma)
        kernelScale = 1 / sqrt(2 * gamma);

        foldAcc = zeros(kfold,1);

        for k = 1:kfold
            idxTrainFold = training(cv, k);
            idxValFold   = test(cv, k);

            Xtr = XtrainStd(idxTrainFold, :);
            Ytr = Ytrain(idxTrainFold, :);
            Xval = XtrainStd(idxValFold, :);
            Yval = Ytrain(idxValFold, :);

            svmModel = fitcsvm(Xtr, Ytr, ...
                "KernelFunction", "rbf", ...
                "KernelScale", kernelScale, ...
                "BoxConstraint", C, ...
                "Standardize", false);

            YvalPred = predict(svmModel, Xval);
            foldAcc(k) = mean(YvalPred == Yval);
        end

        meanAcc = mean(foldAcc);
        fprintf("C = %.3f, gamma = %.4f -> CV Accuracy = %.4f\n", C, gamma, meanAcc);

        if meanAcc > bestAccSVM
            bestAccSVM = meanAcc;
            bestC = C;
            bestGamma = gamma;
            bestKernelScale = kernelScale;
        end
    end
end

fprintf("\nBest SVM: C = %.3f, gamma = %.4f, CV Acc = %.4f\n", ...
    bestC, bestGamma, bestAccSVM);

%% 5. Train Final SVM on Full Training Set

svmModelBest = fitcsvm(XtrainStd, Ytrain, ...
    "KernelFunction", "rbf", ...
    "KernelScale", bestKernelScale, ...
    "BoxConstraint", bestC, ...
    "Standardize", false);

svmModelBest = fitPosterior(svmModelBest);

%% 6. Train a Decision Tree (simple, no hyperparam loop)

treeModelBest = fitctree(Xtrain, Ytrain);   % default settings

%% 7. Choose Positive Class ('M' = malignant)

if any(strcmp(classes, 'M'))
    posLabel = categorical("M");
else
    posLabel = classes{1};
end

%% 8. Evaluate SVM on Test Set

YpredSVM = predict(svmModelBest, XtestStd);
cmSVM = confusionmat(Ytest, YpredSVM);

TP = sum(YpredSVM == posLabel & Ytest == posLabel);
FP = sum(YpredSVM == posLabel & Ytest ~= posLabel);
FN = sum(YpredSVM ~= posLabel & Ytest == posLabel);
TN = sum(YpredSVM ~= posLabel & Ytest ~= posLabel);

accSVM = (TP + TN) / (TP + TN + FP + FN);
precSVM = TP / max(1, TP + FP);
recSVM  = TP / max(1, TP + FN);
if precSVM + recSVM == 0
    f1SVM = 0;
else
    f1SVM = 2 * (precSVM * recSVM) / (precSVM + recSVM);
end

fprintf('\n=== SVM Test Performance ===\n');
fprintf('Accuracy : %.4f\n', accSVM);
fprintf('Precision: %.4f\n', precSVM);
fprintf('Recall   : %.4f\n', recSVM);
fprintf('F1-score : %.4f\n', f1SVM);
disp('Confusion matrix (rows = true, cols = predicted):');
disp(cmSVM);

figure;
confusionchart(Ytest, YpredSVM);
title('SVM Confusion Matrix');

[~, scoresSVM] = predict(svmModelBest, XtestStd);
classOrderSVM = svmModelBest.ClassNames;
posIdx = find(classOrderSVM == posLabel);
scoresPosSVM = scoresSVM(:, posIdx);

[XsvmROC, YsvmROC, ~, AUC_SVM] = perfcurve(Ytest, scoresPosSVM, posLabel);

figure;
plot(XsvmROC, YsvmROC, 'LineWidth', 2); hold on;
plot([0 1], [0 1], '--');
xlabel('False positive rate');
ylabel('True positive rate');
title(sprintf('SVM ROC Curve (AUC = %.3f)', AUC_SVM));
grid on;

%% 9. Evaluate Decision Tree on Test Set

YpredTree = predict(treeModelBest, Xtest);
cmTree = confusionmat(Ytest, YpredTree);

TP = sum(YpredTree == posLabel & Ytest == posLabel);
FP = sum(YpredTree == posLabel & Ytest ~= posLabel);
FN = sum(YpredTree ~= posLabel & Ytest == posLabel);
TN = sum(YpredTree ~= posLabel & Ytest ~= posLabel);

accTree = (TP + TN) / (TP + TN + FP + FN);
precTree = TP / max(1, TP + FP);
recTree  = TP / max(1, TP + FN);
if precTree + recTree == 0
    f1Tree = 0;
else
    f1Tree = 2 * (precTree * recTree) / (precTree + recTree);
end

fprintf('\n=== Decision Tree Test Performance ===\n');
fprintf('Accuracy : %.4f\n', accTree);
fprintf('Precision: %.4f\n', precTree);
fprintf('Recall   : %.4f\n', recTree);
fprintf('F1-score : %.4f\n', f1Tree);
disp('Confusion matrix (rows = true, cols = predicted):');
disp(cmTree);

figure;
confusionchart(Ytest, YpredTree);
title('Decision Tree Confusion Matrix');

[~, scoresTree] = predict(treeModelBest, Xtest);
classOrderTree = treeModelBest.ClassNames;
posIdxTree = find(classOrderTree == posLabel);
scoresPosTree = scoresTree(:, posIdxTree);

[XtreeROC, YtreeROC, ~, AUC_Tree] = perfcurve(Ytest, scoresPosTree, posLabel);

figure;
plot(XtreeROC, YtreeROC, 'LineWidth', 2); hold on;
plot([0 1], [0 1], '--');
xlabel('False positive rate');
ylabel('True positive rate');
title(sprintf('Decision Tree ROC Curve (AUC = %.3f)', AUC_Tree));
grid on;

%% 10. Summary

fprintf('\n=== Summary (Test Set) ===\n');
fprintf('SVM  -> Acc: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f, AUC: %.4f\n', ...
    accSVM, precSVM, recSVM, f1SVM, AUC_SVM);
fprintf('Tree -> Acc: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f, AUC: %.4f\n', ...
    accTree, precTree, recTree, f1Tree, AUC_Tree);
