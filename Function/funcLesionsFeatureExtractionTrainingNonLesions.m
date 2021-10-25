function [m_FeatureManager] = funcLesionsFeatureExtractionTrainingNonLesions(imgStdFLAIR, imgBinary, arrImgLesionCandidate, m_FeatureManager, Level, offsets, iMethod)


for idir = 1:size(imgStdFLAIR, 3)

    a = imgStdFLAIR(:,:,idir);
    b = logical(imgBinary(:,:,idir));
    c = logical(arrImgLesionCandidate(:,:,idir));

    figure(1);imagesc(imgStdFLAIR(:,:,idir));



    roseDcmImage = b;
    binWMLDcmImage = c;


    [roselabelImage numRose] = bwlabel(roseDcmImage);
    figure(101);imagesc(roselabelImage);
    [labelWMLDcmImage numBin] = bwlabel(binWMLDcmImage);
    figure(102);imagesc(labelWMLDcmImage);


    for jter=1:numRose
        tempRose = roselabelImage == jter;

        for kter=1:numBin
            tempBin = labelWMLDcmImage == kter;

            mImage = tempRose.*tempBin;
            sMatchingPercentage = sum(mImage(:))/sum(tempRose(:));
            if sMatchingPercentage > 0.1
                [iy, ix] = find(tempBin);
                binWMLDcmImage(iy, ix) = 0;
                aaa = bwlabel(binWMLDcmImage);
                figure(103);imagesc(aaa);
            end
        end


    end

    b = bwareaopen(binWMLDcmImage, 5);


    [blabel num]= bwlabel(b);
    stats=regionprops(blabel,'BoundingBox');
    for k = 1:numel(stats)
        imgTemp = zeros(size(b));
        imgTemp(blabel == k) =1;


        c=double(a).*double(imgTemp);

        blabelImage = bwlabel(imgTemp);
        stats=regionprops(blabelImage,'BoundingBox');

        if length(stats) == 0
            continue;
        end
        w = stats(1).BoundingBox;
        % Extract sub-image using imcrop():
        subImage = imcrop(c, [w(1), w(2), w(3), w(4)]);
        zeroSize =length(find(subImage == 0));
        m_FeatureCollection = struct();
        histogram_features = FuncChip_histogram_features(subImage,'NumLevels',8192,'G',[ ], 'ZeroLength', zeroSize);
        m_FeatureCollection.Mean = histogram_features(1);
        m_FeatureCollection.Variance = histogram_features(2);
        m_FeatureCollection.Skewness = histogram_features(3);
        m_FeatureCollection.Kurtosis = histogram_features(4);
        m_FeatureCollection.Energy = histogram_features(5);
        m_FeatureCollection.Entropy = histogram_features(6);


        reshapeSubImage = reshape(subImage',1,numel(subImage));
        index = find(reshapeSubImage > 0);
        aMin = double(min(min(reshapeSubImage(index))));
        aMax = double(max(max(reshapeSubImage(index))));

        if iMethod == 1
            r2 = graycomatrix (subImage, 'GrayLimits', [aMin aMax], 'NumLevels', Level, 'Offset',offsets);
            
        elseif iMethod == 2

            %% Please define cluster number for kmean and FCM
            numCluster = Level+1;

            %% K-Mean
            [g,k]=kmeans(subImage(:),numCluster, 'EmptyAction','singleton');
            J = reshape(g,size(subImage));
            %% FCM
            %         options = [NaN NaN NaN 0];
            %         [K, U] = fcm(subImage(:),numCluster, options);%K = cluster_center, g = cluster_idx
            %         [dmp,g] = max(U);
            %         clear dmp; % just a suggestion..
            %         J = reshape(g,size(subImage));

            %% pixel arrangements
            temp = [];
            for iter=1:numCluster
                clear rows, clear cols, clear vals;
                [rows,cols,vals] = find(J == iter);
                val =[];
                for iRun=1:length(rows)
                    val = [val subImage(rows(iRun), cols(iRun))];
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
            % start with label 2 because the label 1 represent background
            r2 = graycomatrix (reLabelImg, 'GrayLimits', [2 numCluster], 'NumLevels',Level, 'Offset',offsets);
            
        end
        stats = graycoprops(r2,{'Contrast','Homogeneity','Energy', 'Correlation'});
        m_FeatureCollection.GLCM_contrast = stats.Contrast;
        m_FeatureCollection.GLCM_homogeneity = stats.Homogeneity;
        m_FeatureCollection.GLCM_Energy = stats.Energy;
        m_FeatureCollection.GLCM_Correlation = stats.Correlation;


        m_FeatureManager = [m_FeatureManager; m_FeatureCollection];

        display([num2str(m_FeatureCollection.Mean), ', ', num2str(m_FeatureCollection.Energy), ', ', num2str(m_FeatureCollection.Entropy), ', Lesions'])

    end
end

end