clear all
close all
clc


dirPath = 'D:\MAT_DATASET_TEST\';
dirFPPath = 'D:\DETECTED_WMLFP\';

dirOutputPath = 'D:\SEGMENTED_WML\';
mkdir(dirOutputPath);
FolderList = dir([dirPath '\*.mat']);

kStep = 5;
minptsub = 20;
minptslb = 3;
theta = 2;
se = strel('disk',3);
load('ReferenceModel.mat');%load variable IntensityReferenceModel

for iDir = 1 : length(FolderList)
    display(['Processing folder :' FolderList(iDir).name]);
  
    load([dirPath FolderList(iDir).name])
    load([dirFPPath FolderList(iDir).name])

    mkdir(dirOutputPath);
    arrImgFinalSegmentationLesions= [];
    for iSelection = 1:length(arrInfo)
        strMetadata = arrInfo(iSelection);
        imgSTD = arrImgStdIntensity(:,:,iSelection);
        imgClassiedWML = arrImgFinalLesions(:,:,iSelection);
        
        if sum(imgClassiedWML(:)) == 0
            arrImgFinalSegmentationLesions(:,:,iSelection) = zeros(size(imgClassiedWML));
            continue;
        end
        
        imgWML = imgClassiedWML == 1;
        imgWML = bwareaopen(imgWML, 5);
        GrayScaleImageWML = double(imgSTD) .* double(imgWML);
        
        if sum(imgWML(:)) == 0
            arrImgFinalSegmentationLesions(:,:,iSelection) = zeros(size(imgClassiedWML));
            continue;
        end
        
        [blabelImageWML num] = bwlabel(imgWML);
        stats=regionprops(blabelImageWML,'BoundingBox');

        imgFinalSegmentedLesions = zeros(size(imgClassiedWML));
        for jter = 1:num
            w = stats(jter).BoundingBox;
            bwImg = blabelImageWML == jter;

            dilatedBW = imdilate(bwImg,se);
            DilateDetailImage = double(imgSTD).*double(dilatedBW);
            subImageWML = imcrop(DilateDetailImage, [w(1), w(2), w(3), w(4)]);
            imgSegmentedWML = funcWMLLOFSegmentation(kStep, minptsub, minptslb, theta, subImageWML, IntensityReferenceModel);
            bwSegmentedWML = imgSegmentedWML ==1;
            imgFinalSegmentedLesions(w(2):w(2)+w(4),w(1):w(1)+w(3)) = bwSegmentedWML;
        end
        
        figure(2);imshow(mat2gray(imgSTD));
        hold on
        
        [B,L] = bwboundaries(imgFinalSegmentedLesions,'noholes');
        for k = 1:length(B)
            position = B{k};
            patch(position(:,2), position(:,1) , [0 1 0], 'FaceColor', [0 1 0] , 'FaceAlpha', 0.5);
        end
        
        [B,L] = bwboundaries(imgWML,'noholes');
        for k = 1:length(B)
            position = B{k};
            patch(position(:,2), position(:,1) , [1 0 0], 'FaceColor', [1 0 0] , 'FaceAlpha', 0.5);
        end
        
        hold off

        set(gca,'position',[0 0 1 1],'units','normalized');
        strStudyID = arrInfo(iSelection).StudyID;
        strPatientID = regexprep(arrInfo(iSelection).PatientID, ' ','_');
        strPatientID = regexprep(strPatientID, '/','_');   
        arrImgFinalSegmentationLesions(:,:,iSelection) = imgFinalSegmentedLesions;
        pause(2);
        
    end
    
    save([dirOutputPath strStudyID '.mat'], 'arrImgFinalSegmentationLesions');
    display([ strStudyID '_' strPatientID '.mat is genearated and saved...']);
end

