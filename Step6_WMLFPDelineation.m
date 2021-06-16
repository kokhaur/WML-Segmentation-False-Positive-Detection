

clear all
close all
clc

%% Initializing
% adding the path to matlab2weka codes
addpath([pwd filesep 'matlab2weka']);
% adding Weka Jar file
if strcmp(filesep, '\')% Windows
        javaaddpath('C:\Program Files (x86)\Weka-3-6\weka.jar');
        
%     javaaddpath('C:\Program Files\Weka-3-6\weka.jar');
elseif strcmp(filesep, '/')% Mac OS X
    javaaddpath('/Applications/weka-3-6-12/weka.jar')
end
% adding matlab2weka JAR file that converts the matlab matrices (and cells)
% to Weka instances.

%% Feature data preparation
numExtraClass = 0;

m_iNumCluster = 5;

dirPath = 'D:\MAT_DATASET_TEST\';
dirOutputPath = 'D:\DETECTED_WMLFP\';

FolderList = dir([dirPath '\*.mat']);

for iDir = 1 : length(FolderList)
    display(['Processing folder :' FolderList(iDir).name]);
  
    load([dirPath FolderList(iDir).name])
    
    mkdir(dirOutputPath);
    arrImgFinalLesions= [];
    for iSelection = 1:length(arrInfo)
        strMetadata = arrInfo(iSelection);
        imgSTD = arrImgStdIntensity(:,:,iSelection);
        imgFPWML = arrImgLesionCandidate(:,:,iSelection);
        
        if sum(imgFPWML(:)) == 0
            arrImgFinalLesions(:,:,iSelection) = zeros(size(imgFPWML));
            continue;
        end
        
        imgWML = imgFPWML == 1;
        imgWML = bwareaopen(imgWML, 5);
        GrayScaleImageWML = double(imgSTD) .* double(imgWML);
        
        if sum(imgWML(:)) == 0
            arrImgFinalLesions(:,:,iSelection) = zeros(size(imgFPWML));
            continue;
        end
        
        [blabelImageWML num] = bwlabel(imgWML);
        stats=regionprops(blabelImageWML,'BoundingBox');
        LesionsFeatureDataset = [];
        for jter = 1:num
            SubItemImageWML = blabelImageWML == jter;
            SubGrayScaleImageWML = SubItemImageWML .* GrayScaleImageWML;
            
            w = stats(jter).BoundingBox;
            if w(3) * w(4)> 100
                a=0;
            end
            subImageWML = imcrop(SubGrayScaleImageWML, [w(1), w(2), w(3), w(4)]);
            LesionsFeature = funcWMLFPDelineationFeatureExtraction(subImageWML, m_iNumCluster);
            LesionsFeatureDataset = [LesionsFeatureDataset; LesionsFeature];
        end
        
        %pick random one lesion or more lesions feature to test
        featTest = LesionsFeatureDataset(:,1:10);
        featName = {'sepal Mean', 'sepal Variance', 'petal Skewness', 'petal Kurtosis' 'sepal Energy', 'sepal Entropy', 'petal GLCM_contrast', 'petal GLCM_homogeneity', 'petal GLCM_Energy', 'petal GLCM_Correlation'};
        classTest =  [{'Non-lesion'},{'Lesion'}];
        %convert the testing data to an Weka object
        display('    Converting Data into WEKA format...');
        ft_test_weka = funcWekaInstance('test', featName, featTest, classTest);
        display('    Converting Completed!');
        
        
        %% read the classification model
        trainModel = weka.core.SerializationHelper.read('randomforest_std_5_onethird.model');

        trainModel.toString()
        
        %% Making Predictions
        
        predicted = cell(ft_test_weka.numInstances()-numExtraClass, 1); %predicted labels
        probDistr = zeros(ft_test_weka.numInstances()-numExtraClass, ft_test_weka.numClasses()); %probability distribution of the predictions
        %the following loop is very slow. We may consider implementing the
        %following in JAVA
        for z = 1:ft_test_weka.numInstances()-numExtraClass
           
            predicted{z,1} = ft_test_weka.instance(z-1).classAttribute.value(trainModel.classifyInstance(ft_test_weka.instance(z-1))).char();% Modified by GM
            
            probDistr(z,:) = (trainModel.distributionForInstance(ft_test_weka.instance(z-1)))';
            display(['probDistr = [' num2str(probDistr(z,1)) ',' num2str(probDistr(z,2)) '], Predicted = ' predicted{z,1}]);
        end
        display('    Classification Completed!');
        
        [blabelImageWML num] = bwlabel(imgWML);
        bwPredictedTrueLesionsImage = zeros(size(imgWML));
        bwPredictedFPImage = zeros(size(imgWML));
        for jter = 1:num
            SubItemImageWML = blabelImageWML == jter;
            if strcmp(predicted{jter,1} , 'Lesion')
                bwPredictedTrueLesionsImage = bwPredictedTrueLesionsImage + SubItemImageWML;% .* GrayScaleImageWML;
            else
                bwPredictedFPImage = bwPredictedFPImage + SubItemImageWML;
            end
        end
        
        figure(2);imshow(mat2gray(imgSTD));
        hold on
        
        [B,L] = bwboundaries(bwPredictedTrueLesionsImage,'noholes');
        for k = 1:length(B)
            position = B{k};
            patch(position(:,2), position(:,1) , [0 1 0], 'FaceColor', [0 1 0] , 'FaceAlpha', 0.5);
        end
        
        [B,L] = bwboundaries(bwPredictedFPImage,'noholes');
        for k = 1:length(B)
            position = B{k};
            patch(position(:,2), position(:,1) , [1 0 0], 'FaceColor', [1 0 0] , 'FaceAlpha', 0.5);
        end
        
        hold off

        set(gca,'position',[0 0 1 1],'units','normalized');
        strStudyID = arrInfo(iSelection).StudyID;
        strPatientID = regexprep(arrInfo(iSelection).PatientID, ' ','_');
        strPatientID = regexprep(strPatientID, '/','_');
        
        
        TPFPImage = bwPredictedTrueLesionsImage + 2.*bwPredictedFPImage;
        arrImgFinalLesions(:,:,iSelection) = TPFPImage;
    
        pause(2);
        
    end
    
    save([dirOutputPath strStudyID '.mat'], 'arrImgLesionCandidate','arrImgFinalLesions');
    display([ strStudyID '_' strPatientID '.mat is genearated and saved...']);
end
