
function funcFPFeatureErrorEvaluation(folderPath)

close all
clc

m_evalCollection = [];
BeginCluster = 2;
NoCluster = 8;

for iLevel=BeginCluster:NoCluster
clearvars -except iLevel m_evalCollection folderPath NoCluster BeginCluster; close all;

%% Initializing 
% adding the path to matlab2weka codes
addpath([pwd filesep 'matlab2weka']);
% adding Weka Jar file
if strcmp(filesep, '\')% Windows   
    
%     javaaddpath('C:\Program Files\Weka-3-6\weka.jar');
    javaaddpath('C:\Program Files (x86)\Weka-3-6\weka.jar');
elseif strcmp(filesep, '/')% Mac OS X
    javaaddpath('/Applications/weka-3-6-12/weka.jar')
end
% adding matlab2weka JAR file that converts the matlab matrices (and cells)
% to Weka instances.
javaaddpath([pwd filesep 'matlab2weka' filesep 'matlab2weka.jar']);

%% Loading Dataset
load([folderPath 'LesionsFeatureDataset_lvl_',num2str(iLevel)])

% numerical class variable
feat_num = LesionsFeatureDataset(:,1:10);
featName = {'sepal Mean',...
            'sepal Variance',...
            'petal Skewness',...
            'petal Kurtosis',...
            'sepal Energy',...
            'sepal Entropy',...
            'petal GLCM_contrast',...
            'petal GLCM_homogeneity',...
            'petal GLCM_Energy',...
            'petal GLCM_Correlation'...
            };
class_num = LesionsFeatureDataset(:,11);

% converting to nominal variables (Weka cannot classify numerical classes)
class_nom = cell(size(class_num));
uClass_num = unique(class_num);
tmp_cell = cell(1,1);
for i = 1:length(uClass_num)
    tmp_cell{1,1} = strcat('class_', num2str(i-1));
    class_nom(class_num == uClass_num(i),:) = repmat(tmp_cell, sum(class_num == uClass_num(i)), 1);
end
clear uClass_num tmp_cell i 

% Choosing a regression tool to be used
% -------------------------------------
% classifier = 1: Random Forest Classifier from WEKA
% classifier = 2: Gaussian Process Regression from WEKA
% classifier = 3: Support Vector Machine from WEKA
% classifier = 4: Logistic Regression from WEKA
classifier = 1;

%% Performing K-fold Cross Validation
K = 10;
N = size(feat_num,1);
%indices for cross validation
idxCV = ceil(rand([1 N])*K); 
actualClass = cell(size(feat_num,1),1);
predictedClass = cell(size(feat_num,1),1);
for k = 1:K
    %defining training and testing sets
    feature_train = feat_num(idxCV ~= k,:);
    class_train = class_nom(idxCV ~= k,1);
    feature_test = feat_num(idxCV == k,:);
    class_test = class_nom(idxCV == k,1);
    
    %performing regression
    [actual_tmp, predicted_tmp, probDistr_tmp] = wekaClassification(feature_train, class_train, feature_test, class_test, featName, classifier);
    
    %accumulating the results
    actualClass(idxCV == k,:) = actual_tmp;
    predictedClass(idxCV == k,:) = predicted_tmp;    
    clear feature_train class_train feature_test class_test
    clear actual_tmp predicted_tmp probDistr_tmp
end
clear idxCV k
%% Performing K-fold Cross Validation using weka default crossvalidation
%% evaluation
m_evalObject = wekaCrossValidation(feat_num, class_nom, featName, classifier);

m_evalCollection = [m_evalCollection; m_evalObject];
end

%% plot output results
m_pctCorrect = [];
m_pctIncorrect = [];
m_weightedFalsePositiveRate = [];
m_weightedTruePositiveRate = [];
m_weightedTrueNegativeRate = [];
m_weightedFalseNegativeRate = [];
m_weightedPrecision = [];
m_weightedRecall = [];

for iter = 1 : length(m_evalCollection)
    m_pctCorrect = [m_pctCorrect m_evalCollection(iter).pctCorrect];
    m_weightedTruePositiveRate = [m_weightedTruePositiveRate m_evalCollection(iter).weightedTruePositiveRate];
    m_weightedTrueNegativeRate = [m_weightedTrueNegativeRate m_evalCollection(iter).weightedTrueNegativeRate];
    m_weightedFalsePositiveRate = [m_weightedFalsePositiveRate m_evalCollection(iter).weightedFalsePositiveRate];
    m_weightedPrecision = [m_weightedPrecision m_evalCollection(iter).weightedPrecision];
    m_weightedRecall = [m_weightedRecall m_evalCollection(iter).weightedRecall];
end

h1 = figure(10); 
ycoor = BeginCluster:NoCluster;
plot(ycoor,m_pctCorrect/100,'-r+', ycoor, m_weightedTruePositiveRate,  '--g.',  ycoor, m_weightedPrecision, ':bs', ycoor, m_weightedRecall, ':bo',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])
saveas(h1, [folderPath 'AccuracyMix.fig']);

h2 = figure(11); 
labels = cellstr( num2str([BeginCluster:NoCluster]') );
plot(m_weightedFalsePositiveRate*100, m_weightedTruePositiveRate*100, 'go',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])
text(m_weightedFalsePositiveRate*100, m_weightedTruePositiveRate*100, labels, 'VerticalAlignment','bottom', ...
                             'HorizontalAlignment','right')
axis([0 100 0 100])
ylabel('True positive rate (Sensitivity)')
xlabel('Flase positive rate (1-specificity)')
saveas(h2, [folderPath 'SenvSpec.fig']);

h3 = figure(12); 
ycoor = BeginCluster:NoCluster;
plot(ycoor,m_pctCorrect/100,'-r+', ...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])
saveas(h3, [folderPath 'AccuracyPredictCorrection.fig']);

iIndex=  strfind(folderPath,'\');
TempFileName = folderPath(iIndex(3)+1:length(folderPath)-1);
FileName = strrep(TempFileName, '\', '_');
save([folderPath FileName '.mat'],'m_pctCorrect', 'm_weightedTruePositiveRate', 'm_weightedTrueNegativeRate', 'm_weightedFalsePositiveRate', 'm_weightedPrecision', 'm_weightedRecall');


