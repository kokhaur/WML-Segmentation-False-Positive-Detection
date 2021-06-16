function BuildRandomForestModel()
clear all
close all
clc


% FilePath = 'D:\ClusterGLCMExtractedFeatures\kMeans\Angle_0\LesionsFeatureDataset_lvl_5.mat';
FilePath = 'LesionsFeatureDataset_lvl_5.mat';

%% Initializing
% adding the path to matlab2weka codes
addpath([pwd filesep 'matlab2weka']);
% adding Weka Jar file
if strcmp(filesep, '\')% Windows
    javaaddpath('C:\Program Files (x86)\Weka-3-6\weka.jar');
%       javaaddpath('C:\Program Files\Weka-3-6\weka.jar');
elseif strcmp(filesep, '/')% Mac OS X
    javaaddpath('/Applications/weka-3-6-12/weka.jar')
end
% adding matlab2weka JAR file that converts the matlab matrices (and cells)
% to Weka instances.
javaaddpath([pwd filesep 'matlab2weka' filesep 'matlab2weka.jar']);

%% Loading Dataset
load(FilePath);

a = length(find(LesionsFeatureDataset(:,11)==1));
b = length(LesionsFeatureDataset);
% r = a + (b-a).*rand(a+a/2,1);
r = a + (b-a).*rand(a+a/3,1);

Nonlesions = LesionsFeatureDataset(uint16(r),:);
Lesions = LesionsFeatureDataset(1:a,:);
LesionsFeatureDataset =[];
LesionsFeatureDataset = [Lesions; Nonlesions];
% numerical class variable
feat_num = LesionsFeatureDataset(:,1:10);
featName = {'sepal Mean', 'sepal Variance', 'petal Skewness', 'petal Kurtosis' 'sepal Energy', 'sepal Entropy', 'petal GLCM_contrast', 'petal GLCM_homogeneity', 'petal GLCM_Energy', 'petal GLCM_Correlation'};
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





import matlab2weka.*;

%% Converting to WEKA data
display('    Converting Data into WEKA format...');
classTrain = class_nom;
featTrain = feat_num;
%convert the testing data to an Weka object
convert2wekaObj = convert2weka('training', featName, featTrain', classTrain, true);
ft_train_weka = convert2wekaObj.getInstances();
clear convert2wekaObj;
display('    Converting Completed!');


import weka.classifiers.trees.RandomForest.*;
import weka.classifiers.meta.Bagging.*;
%create an java object
trainModel = weka.classifiers.trees.RandomForest();
%defining parameters
trainModel.setMaxDepth(0); %Set the maximum depth of the tree, 0 for unlimited.
trainModel.setNumFeatures(0); %Set the number of features to use in random selection.
trainModel.setNumTrees(100); %Set the value of numTrees.
trainModel.setSeed(1);
%train the classifier
trainModel.buildClassifier(ft_train_weka);
weka.core.SerializationHelper.write('randomforest_std_5_onethird.model', trainModel);