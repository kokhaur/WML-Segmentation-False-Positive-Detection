clear all
close all
clc


outputFolderPath = 'D:\ClusterGLCMExtractedFeatures\';
strPath = 'D:\MAT_DATASET_TRAIN\';
%Four different degree
offsets = [0 1; -1 1;-1 0;-1 -1];

for iApproach = 1 : 2
    iMethod = iApproach;
    if iMethod ==1
        strMethodFolder = 'Quantile';
        mkdir([outputFolderPath strMethodFolder]);
    elseif iMethod == 2
         strMethodFolder = 'kMeans';
        mkdir([outputFolderPath strMethodFolder]);
    end
for iOffSet = 1 : length(offsets)

    if iOffSet == 1
        strSaveFolder = 'Angle_0';
        mkdir([outputFolderPath strMethodFolder '/' strSaveFolder]);
    elseif iOffSet == 2
        strSaveFolder = 'Angle_45';
        mkdir([outputFolderPath strMethodFolder '/' strSaveFolder]);
    elseif iOffSet == 3
        strSaveFolder = 'Angle_90';
        mkdir([outputFolderPath strMethodFolder '/' strSaveFolder]);
    elseif iOffSet == 4
        strSaveFolder = 'Angle_135';
        mkdir([outputFolderPath strMethodFolder '/' strSaveFolder]);
    end

    for iLevel=2:8
        clearvars -except iLevel outputFolderPath offsets strSaveFolder strMethodFolder iOffSet iApproach iMethod strPath
        close all


        clc

        entropyCollection = [];
        meanCollection = [];
        stdCollection = [];

        m_arrFeatureEntropy = [];
        m_arrFeatureMean = [];
        m_arrFeatureVariance = [];
        m_arrFeatureKurtosis = [];
        m_arrFeatureSkewness = [];
        m_arrFeatureEnergy = [];

        m_FeatureManager = [];

        %   Mean
        %   Variance
        %   Skewness
        %   Kurtosis
        %   Energy
        %   Entropy


        dataClassCollection = [];

        %% create train data structure
        train_dataWeka = struct();
        train_relname = sprintf('train_dataset_%s', datestr(now,'yyyymmdd'));
        train_outfile = sprintf('%s.arff', train_relname);

        % Create dataset in mat format
        LesionsFeatureDataset = [];
        LesionsFeatures = [];
        
        % nominal classes
        type_class = { 'Non-lesions', 'Lesions' };
        iter= 1;

        display('   Training - Lesions   ');

        
        FileList = dir([strPath '*.mat']);

        for i=1:length(FileList)
            load([strPath FileList(i).name]);
            m_FeatureManager = [];
            [m_FeatureManager] = funcLesionsFeatureExtractionTrainingLesions(arrImgStdIntensity, arrImgAnnotated, m_FeatureManager, iLevel, offsets(iOffSet,:), iMethod);

            for iIndex = 1 : length(m_FeatureManager)
               
                train_dataWeka(iter).Mean = m_FeatureManager(iIndex).Mean;
                train_dataWeka(iter).Variance = m_FeatureManager(iIndex).Variance;
                train_dataWeka(iter).Skewness = m_FeatureManager(iIndex).Skewness;
                train_dataWeka(iter).Kurtosis = m_FeatureManager(iIndex).Kurtosis;
                train_dataWeka(iter).Energy = m_FeatureManager(iIndex).Energy;
                train_dataWeka(iter).Entropy = m_FeatureManager(iIndex).Entropy;
                train_dataWeka(iter).GLCM_contrast = m_FeatureManager(iIndex).GLCM_contrast;
                train_dataWeka(iter).GLCM_homogeneity = m_FeatureManager(iIndex).GLCM_homogeneity;
                train_dataWeka(iter).GLCM_Energy = m_FeatureManager(iIndex).GLCM_Energy;
                train_dataWeka(iter).GLCM_Correlation = m_FeatureManager(iIndex).GLCM_Correlation;

                train_dataWeka(iter).type_class = 'Lesions';

                LesionsFeature =[double(train_dataWeka(iter).Mean)...
                    double(train_dataWeka(iter).Variance)...
                    double(train_dataWeka(iter).Skewness)...
                    double(train_dataWeka(iter).Kurtosis)...
                    double(train_dataWeka(iter).Energy)...
                    double(train_dataWeka(iter).Entropy)...
                    double(train_dataWeka(iter).GLCM_contrast)...
                    double(train_dataWeka(iter).GLCM_homogeneity)...
                    double(train_dataWeka(iter).GLCM_Energy)...
                    double(train_dataWeka(iter).GLCM_Correlation)...
                    1.0
                    ];

                LesionsFeatureDataset = [LesionsFeatureDataset; LesionsFeature];


                iter = iter+1;
            end

         
        end


        m_FeatureManager = [];
        display('   Training - Non-lesions   ');
        for i=1:length(FileList)
            load([strPath FileList(i).name]);
            m_FeatureManager = [];
            [m_FeatureManager] = funcLesionsFeatureExtractionTrainingNonLesions(arrImgStdIntensity, arrImgAnnotated, arrImgLesionCandidate, m_FeatureManager, iLevel, offsets(iOffSet,:), iMethod);

            for iIndex = 1 : length(m_FeatureManager)
             
                train_dataWeka(iter).Mean = m_FeatureManager(iIndex).Mean;
                train_dataWeka(iter).Variance = m_FeatureManager(iIndex).Variance;
                train_dataWeka(iter).Skewness = m_FeatureManager(iIndex).Skewness;
                train_dataWeka(iter).Kurtosis = m_FeatureManager(iIndex).Kurtosis;
                train_dataWeka(iter).Energy = m_FeatureManager(iIndex).Energy;
                train_dataWeka(iter).Entropy = m_FeatureManager(iIndex).Entropy;
                train_dataWeka(iter).GLCM_contrast = m_FeatureManager(iIndex).GLCM_contrast;
                train_dataWeka(iter).GLCM_homogeneity = m_FeatureManager(iIndex).GLCM_homogeneity;
                train_dataWeka(iter).GLCM_Energy = m_FeatureManager(iIndex).GLCM_Energy;
                train_dataWeka(iter).GLCM_Correlation = m_FeatureManager(iIndex).GLCM_Correlation;

                train_dataWeka(iter).type_class = 'Non-lesions';

                LesionsFeature =[double(train_dataWeka(iter).Mean)...
                    double(train_dataWeka(iter).Variance)...
                    double(train_dataWeka(iter).Skewness)...
                    double(train_dataWeka(iter).Kurtosis)...
                    double(train_dataWeka(iter).Energy)...
                    double(train_dataWeka(iter).Entropy)...
                    double(train_dataWeka(iter).GLCM_contrast)...
                    double(train_dataWeka(iter).GLCM_homogeneity)...
                    double(train_dataWeka(iter).GLCM_Energy)...
                    double(train_dataWeka(iter).GLCM_Correlation)...
                    0.0
                    ];

                LesionsFeatureDataset = [LesionsFeatureDataset; LesionsFeature];


                iter = iter+1;
            end

            display('    ');
        end

       

        %% declare nominal specification attributes
        nomspec.type_class = type_class;

        % save arff
        % arff_write(train_outfile, train_dataWeka, train_relname, nomspec);

        %save in mat file
        save([outputFolderPath strMethodFolder '/' strSaveFolder '/LesionsFeatureDataset_lvl_',num2str(iLevel),'.mat'], 'LesionsFeatureDataset');
    end

end

end
