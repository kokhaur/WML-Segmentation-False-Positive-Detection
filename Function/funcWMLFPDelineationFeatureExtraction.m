function LesionsFeature = funcWMLFPDelineationFeatureExtraction(WMLPatch, iLevel)

        numCluster = iLevel+1;
        
        m_FeatureCollection = struct();
        zeroSize =length(find(WMLPatch == 0));
        histogram_features = FuncChip_histogram_features(WMLPatch,'NumLevels',8192,'G',[ ], 'ZeroLength', zeroSize);
        m_FeatureCollection.Mean = histogram_features(1);
        m_FeatureCollection.Variance = histogram_features(2);
        m_FeatureCollection.Skewness = histogram_features(3);
        m_FeatureCollection.Kurtosis = histogram_features(4);
        m_FeatureCollection.Energy = histogram_features(5);
        m_FeatureCollection.Entropy = histogram_features(6);
        
        
        [g,k]=kmeans(WMLPatch(:),numCluster, 'EmptyAction','singleton');
        J = reshape(g,size(WMLPatch));
        
         %% pixel arrangements
        temp = [];
        for iter=1:numCluster
            clear rows, clear cols, clear vals;
            [rows,cols,vals] = find(J == iter);
            val =[];
            for iRun=1:length(rows)
                val = [val WMLPatch(rows(iRun), cols(iRun))];
            end
            temp = [temp max(val)];

        end

        [B,Index] = sort(temp);
        reLabelImg = zeros(size(J));

         


        for iter=1:numCluster
            [rows,cols,vals] = find(J == Index(iter));
            for iRun=1:length(rows)
                reLabelImg(rows(iRun), cols(iRun)) = iter;
            end
        end

%         [reLabelImg,C,U,LUT,H]=FastFCMeans(uint16(WMLPatch),numCluster);

        r2 = graycomatrix (reLabelImg, 'GrayLimits', [2 numCluster], 'NumLevels',iLevel);
        stats = graycoprops(r2,{'contrast','homogeneity','Energy', 'Correlation'});


        m_FeatureCollection.GLCM_contrast = stats.Contrast;
        m_FeatureCollection.GLCM_homogeneity = stats.Homogeneity;
        m_FeatureCollection.GLCM_Energy = stats.Energy;
        m_FeatureCollection.GLCM_Correlation = stats.Correlation;


        LesionsFeature =[double(m_FeatureCollection.Mean)...
            double(m_FeatureCollection.Variance)...
            double(m_FeatureCollection.Skewness)...
            double(m_FeatureCollection.Kurtosis)...
            double(m_FeatureCollection.Energy)...
            double(m_FeatureCollection.Entropy)...
            double(m_FeatureCollection.GLCM_contrast)...
            double(m_FeatureCollection.GLCM_homogeneity)...
            double(m_FeatureCollection.GLCM_Energy)...
            double(m_FeatureCollection.GLCM_Correlation)...
            0.0
            ];

        

        
