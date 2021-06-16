clear all
close all
clc

strPath = 'D:\DICOM_INPUT_TEST\';
strSavePath = 'D:\MAT_DATASET_TEST\';
FolderList = dir(strPath);

for iter = 3 : length(FolderList)
    tf = isdir([strPath FolderList(iter).name]);
    
    if ~tf
        continue;
    end
    
    strFLAIRFolderName = [FolderList(iter).name '\InMEcJ\FLAIR\'];
    strBrainClassificationFolderName = [FolderList(iter).name '\InMEcJ\BrainClassification\'];
    strStdIntensityFolderName = [FolderList(iter).name '\InMEcJ\MRIntensityStandardisation\'];
    strLesionCandidateFolderName = [FolderList(iter).name '\InMEcJ\WMLesionSegmentation\'];
    
    ImageList = dir([strPath strFLAIRFolderName '\*.dcm']);
    
    arrInfo = [];
    arrImgFLAIR = [];
    arrImgBrain = [];
    arrImgStdIntensity = [];
    arrImgLesionCandidate = [];
    info = dicominfo([strPath strFLAIRFolderName ImageList(1).name]);
    strStudyID = info.StudyID;
    for jter = 1 : length(ImageList)
        strFileName = ImageList(jter).name;
        
        info = dicominfo([strPath strFLAIRFolderName strFileName]);
        imgFLAIR = dicomread([strPath strFLAIRFolderName strFileName]);
        imgBrain = dicomread([strPath strBrainClassificationFolderName strFileName]);
        imgStdIntensity = dicomread([strPath strStdIntensityFolderName strFileName]);
        imgLesionCandidate = dicomread([strPath strLesionCandidateFolderName strFileName]);
        
        arrInfo = [arrInfo; info];
        arrImgBrain(:,:,jter) = imgBrain;
        arrImgFLAIR(:,:,jter) = imgFLAIR;
        arrImgStdIntensity(:,:,jter) = imgStdIntensity;
        arrImgLesionCandidate(:,:,jter) = imgLesionCandidate;

    end
%     arrImgFLAIR = imrotate(arrImgFLAIR,180);
    save([strSavePath strStudyID '.mat'], 'arrInfo', 'arrImgFLAIR', 'arrImgBrain', 'arrImgStdIntensity', 'arrImgLesionCandidate');
    display([ strStudyID '.mat is genearated and saved...']);
end
