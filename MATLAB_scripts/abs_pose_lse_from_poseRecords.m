
function abs_pose_lse_from_poseRecords(repoRoot, resultsDir, iouMatFile)
%% ============================================================
% CONFIGURATION
%% ============================================================

config = struct();
config.repoRoot   = repoRoot;
config.resultsDir = resultsDir;

if ~exist(config.resultsDir,'dir'), mkdir(config.resultsDir); end

config.iouMatFile = iouMatFile;

% config = struct();
% 
% config.repoRoot   = fileparts(mfilename('fullpath'));
% config.resultsDir = fullfile(config.repoRoot, "results");
% if ~exist(config.resultsDir,'dir'), mkdir(config.resultsDir); end
% config.iouMatFile = fullfile(config.resultsDir, "VehBuildMinimal_poseRecords_for_bestpose.mat");

addpath(genpath(fullfile(config.repoRoot, "Functions")));
addpath(genpath(fullfile(config.repoRoot, "rotation_error")));


%% ============================================================
% LOAD INPUT DATA
%% ============================================================

fprintf('[INFO] Loading input MAT file: %s\n', config.iouMatFile);
load(config.iouMatFile);   % loads poseRecords

allObjectHypotheses = poseRecords;

fprintf('[INFO] Loaded %d rows from poseRecords\n', numel(allObjectHypotheses));

%% ============================================================
% GLOBAL PREALLOCATION
%% ============================================================

sinSqThreshold = sind(5)^2;

summaryRows = {};
summaryRowCount = 0;

numImagesWithLessThanTwoInliers = 0;

imageBestPose = struct();
imageBestPoseCount = [];
imageBestPoseAllCount = [];
imageBestPoseInlierIouMean = [];
imageBestPoseDirAngleErrMean = [];

poseRecordsRaw = struct( ...
    'imageName', {}, ...
    'semanticId', {}, ...
    'instanceId', {}, ...
    'ellipsoidRadiusX', {}, ...
    'ellipsoidRadiusY', {}, ...
    'ellipsoidRadiusZ', {}, ...
    'ellipsoidRotationMatrix', {}, ...
    'ellipsoidCenter', {}, ...
    'cameraIntrinsicMatrix', {}, ...
    'cameraCenterWorld', {}, ...
    'estimatedRotationCamFromWorld', {}, ...
    'translationErrorMeters', {}, ...
    'rotationErrorDegrees', {}, ...
    'ellipseRotationMatrix2D', {}, ...
    'ellipseSemiMajorPixels', {}, ...
    'ellipseSemiMinorPixels', {}, ...
    'ellipseCenterPixels', {}, ...
    'predictedDirectionX', {}, ...
    'predictedDirectionY', {}, ...
    'groundTruthDirectionX', {}, ...
    'groundTruthDirectionY', {}, ...
    'groundTruthPoseHomogeneous', {}, ...
    'rotationXCam', {}, ...
    'rotationYCam', {}, ...
    'rotationZCam', {}, ...
    'rotationZEstimated', {} );

poseRecordCount = 0;

%% ============================================================
% SPLIT BY SEQUENCE PREFIX IN imageName
%% ============================================================

allImageNames = string({allObjectHypotheses.imageName})';

if any(~contains(allImageNames, "_"))
    error('Some imageName values do not contain "_" separator. Expected format like "09_0000000113.png".');
end

sequenceIdStrings = extractBefore(allImageNames, "_");
sequenceIds = str2double(sequenceIdStrings);

if any(isnan(sequenceIds))
    error('Could not parse sequence ID from some imageName values. Expected prefix like "09_" or "10_".');
end

uniqueSequences = unique(sequenceIds, 'stable');

fprintf('[INFO] Found %d sequences: ', numel(uniqueSequences));
fprintf('%d ', uniqueSequences);
fprintf('\n');

%% ============================================================
% MAIN LOOP OVER SEQUENCES
%% ============================================================

for sequenceLoopIndex = 1:numel(uniqueSequences)

    currentSequenceId = uniqueSequences(sequenceLoopIndex);
    fprintf('\n============================================================\n');
    fprintf('[SEQ %02d] Processing sequence %02d (%d / %d)\n', ...
        currentSequenceId, currentSequenceId, sequenceLoopIndex, numel(uniqueSequences));
    fprintf('============================================================\n');

    currentSequenceMask = (sequenceIds == currentSequenceId);
    sequenceObjectHypotheses = allObjectHypotheses(currentSequenceMask);

    if isempty(sequenceObjectHypotheses)
        fprintf('[SEQ %02d] No rows found. Skipping.\n', currentSequenceId);
        continue;
    end

    %% ============================================================
    % IMAGE GROUPING WITHIN CURRENT SEQUENCE
    %% ============================================================

    uniqueImageNames = unique({sequenceObjectHypotheses.imageName}');
    numUniqueImages = numel(uniqueImageNames);

    imageBestPose = struct();
    imageBestPose(numUniqueImages).image = [];

    imageBestPoseCount = zeros(numUniqueImages,1);
    imageBestPoseAllCount = zeros(numUniqueImages,1);
    imageBestPoseInlierIouMean = zeros(numUniqueImages,1);
    imageBestPoseDirAngleErrMean = zeros(numUniqueImages,1);

    perImageTimingImageId = strings(numUniqueImages,1);
    perImageTimingSec = nan(numUniqueImages,1);

    fprintf('[SEQ %02d] Number of unique images: %d\n', currentSequenceId, numUniqueImages);

    %% ============================================================
    % MAIN LOOP OVER IMAGES IN CURRENT SEQUENCE
    %% ============================================================

    for imageIndex = 1:numUniqueImages

        currentImageName = uniqueImageNames{imageIndex};
        perImageTimingImageId(imageIndex) = string(currentImageName);

        matchingRows = strcmp({sequenceObjectHypotheses.imageName}', currentImageName);
        imageObjectRows = sequenceObjectHypotheses(matchingRows);

        if numel(imageObjectRows) < 2
            continue;
        end

        poseCandidates = cell(0,1);
        for poseIndex = 1:numel(imageObjectRows)
            poseCandidates{end+1} = struct( ...
                'T',      imageObjectRows(poseIndex).translationMatrixT, ...
                'R',      imageObjectRows(poseIndex).rotationMatrixR, ...
                'K',      imageObjectRows(poseIndex).cameraIntrinsicMatrix, ...
                'Rz',     imageObjectRows(poseIndex).Rz, ...
                'Rz_est', imageObjectRows(poseIndex).Rz_est );
        end

        bestPose = [];
        bestPoseInlierCount = -1;
        bestPoseMeanSinSq = Inf;

        bestInlierObjects = [];
        allObjectsOnImage = [];

        minRotationErrorThisImage = Inf;
        maxRotationErrorThisImage = -Inf;

        bestMeanIoU = NaN;
        bestMeanDirectionAngleErrorDeg = NaN;

        for poseIndex = 1:numel(poseCandidates)

            selectedPose = poseCandidates{poseIndex};

            inlierCount = 0;
            currentInlierObjects = [];
            currentAllObjects = [];

            sinSqPerObject = Inf(numel(imageObjectRows),1);
            dotPerObject = -Inf(numel(imageObjectRows),1);
            dirAngleErrDegPerObject = nan(numel(imageObjectRows),1);

            for objectIndex = 1:numel(imageObjectRows)

                predDirX = imageObjectRows(objectIndex).directionX;
                predDirY = imageObjectRows(objectIndex).directionY;
                gtDirX   = imageObjectRows(objectIndex).GTdirectionX;
                gtDirY   = imageObjectRows(objectIndex).GTdirectionY;
                gtRz     = imageObjectRows(objectIndex).Rz;

                vc_in_world = selectedPose.Rz_est * [predDirX; predDirY; 0];
                vw_gt       = gtRz * [gtDirX; gtDirY; 0];

                vCross = cross(vc_in_world, vw_gt);
                sinTheta = norm(vCross) / (norm(vc_in_world) * norm(vw_gt));
                sinSqTheta = sinTheta^2;

                sinSqPerObject(objectIndex) = sinSqTheta;
                dirAngleErrDegPerObject(objectIndex) = atan2d(norm(vCross), dot(vc_in_world, vw_gt));
                dotPerObject(objectIndex) = dot(vc_in_world, vw_gt);

                if imageObjectRows(objectIndex).rotationError < minRotationErrorThisImage
                    minRotationErrorThisImage = imageObjectRows(objectIndex).rotationError;
                end
                if imageObjectRows(objectIndex).rotationError > maxRotationErrorThisImage
                    maxRotationErrorThisImage = imageObjectRows(objectIndex).rotationError;
                end
            end

            sumSinSqInliers = 0;
            inlierIouList = [];
            inlierDirAngleErrList = [];

            for objectIndex = 1:numel(imageObjectRows)

                if sinSqPerObject(objectIndex) <= sinSqThreshold && dotPerObject(objectIndex) > 0

                    sumSinSqInliers = sumSinSqInliers + sinSqPerObject(objectIndex);
                    inlierCount = inlierCount + 1;
                    currentInlierObjects = [currentInlierObjects; imageObjectRows(objectIndex)]; 

                    cameraModel = createCamera(selectedPose.T, selectedPose.R, selectedPose.K);

                    ellipsoidModel = createEllipsoid( ...
                        imageObjectRows(objectIndex).ellipsoidCenter, ...
                        imageObjectRows(objectIndex).ellipsoidRotationMatrix, ...
                        imageObjectRows(objectIndex).ellipsoidRadiusX, ...
                        imageObjectRows(objectIndex).ellipsoidRadiusY, ...
                        imageObjectRows(objectIndex).ellipsoidRadiusZ);

                    projectedEllipseStruct = projectEllipsoid(ellipsoidModel, cameraModel);

                    projectedEllipseVec = [ ...
                        projectedEllipseStruct.center(1), ...
                        projectedEllipseStruct.center(2), ...
                        projectedEllipseStruct.ai, ...
                        projectedEllipseStruct.bi, ...
                        atan2(projectedEllipseStruct.Ri_el(2,1), projectedEllipseStruct.Ri_el(1,1))];

                    detectedEllipseVec = [ ...
                        imageObjectRows(objectIndex).ellipseCenter(1), ...
                        imageObjectRows(objectIndex).ellipseCenter(2), ...
                        imageObjectRows(objectIndex).ellipseSemiMajorAxis, ...
                        imageObjectRows(objectIndex).ellipseSemiMinorAxis, ...
                        atan2(imageObjectRows(objectIndex).ellipseRotationMatrix(2,1), ...
                              imageObjectRows(objectIndex).ellipseRotationMatrix(1,1))];

                    iouVal = calc_probiou(detectedEllipseVec, projectedEllipseVec);
                    inlierIouList = [inlierIouList; iouVal]; 
                    inlierDirAngleErrList = [inlierDirAngleErrList; dirAngleErrDegPerObject(objectIndex)]; 
                end

                currentAllObjects = [currentAllObjects; imageObjectRows(objectIndex)]; 
            end

            if inlierCount > 0
                currentMeanSinSq = sumSinSqInliers / inlierCount;
            else
                currentMeanSinSq = Inf;
            end

            if inlierCount >= bestPoseInlierCount && currentMeanSinSq < bestPoseMeanSinSq
                bestPoseInlierCount = inlierCount;
                bestPoseMeanSinSq = currentMeanSinSq;
                bestPose = selectedPose;
                bestInlierObjects = currentInlierObjects;
                allObjectsOnImage = currentAllObjects;
                bestMeanIoU = mean(inlierIouList);
                bestMeanDirectionAngleErrorDeg = mean(inlierDirAngleErrList);
            end
        end

        imageBestPose(imageIndex).image = currentImageName;
        imageBestPose(imageIndex).best_pose = bestPose;
        imageBestPose(imageIndex).min_rot_per_img = minRotationErrorThisImage;
        imageBestPose(imageIndex).max_rot_per_img = maxRotationErrorThisImage;
        imageBestPose(imageIndex).inlier_objects = bestInlierObjects;
        imageBestPose(imageIndex).all_objects_onImg = allObjectsOnImage;

        imageBestPoseCount(imageIndex) = bestPoseInlierCount;
        imageBestPoseAllCount(imageIndex) = size(allObjectsOnImage,1);
        imageBestPoseInlierIouMean(imageIndex) = bestMeanIoU;
        imageBestPoseDirAngleErrMean(imageIndex) = bestMeanDirectionAngleErrorDeg;

        fprintf('[SEQ %02d] Best pose for image %s: %d inlier objects.\n', ...
            currentSequenceId, currentImageName, bestPoseInlierCount);

        if numel(imageBestPose(imageIndex).inlier_objects) < 2
            numImagesWithLessThanTwoInliers = numImagesWithLessThanTwoInliers + 1;
        end

        poseSolveStart = tic;

        if numel(imageBestPose(imageIndex).inlier_objects) > 1

            cPolynomialSum = 0;

            for objectIndex = 1:numel(imageBestPose(imageIndex).inlier_objects)

                predDirX = imageBestPose(imageIndex).inlier_objects(objectIndex).directionX;
                predDirY = imageBestPose(imageIndex).inlier_objects(objectIndex).directionY;
                gtDirX   = imageBestPose(imageIndex).inlier_objects(objectIndex).GTdirectionX;
                gtDirY   = imageBestPose(imageIndex).inlier_objects(objectIndex).GTdirectionY;
                gtRz     = imageBestPose(imageIndex).inlier_objects(objectIndex).Rz;

                vc = [predDirX; predDirY; 0];
                vw = gtRz * [gtDirX; gtDirY; 0];

                c = [-4 * vc(1)^2 * vw(1) * vw(2) + 4 * vc(1) * vc(2) * vw(1)^2 - 4 * vc(1) * vc(2) * vw(2)^2 + 4 * vc(2)^2 * vw(1) * vw(2); ...
                      4 * vc(1)^2 * vw(2)^2 - 8 * vc(1) * vc(2) * vw(1) * vw(2) + 4 * vc(2)^2 * vw(1)^2; ...
                      12 * vc(1)^2 * vw(1) * vw(2) - 12 * vc(1) * vc(2) * vw(1)^2 + 12 * vc(1) * vc(2) * vw(2)^2 - 12 * vc(2)^2 * vw(1) * vw(2); ...
                      8 * vc(1)^2 * vw(1)^2 - 4 * vc(1)^2 * vw(2)^2 + 24 * vc(1) * vc(2) * vw(1) * vw(2) - 4 * vc(2)^2 * vw(1)^2 + 8 * vc(2)^2 * vw(2)^2];

                cPolynomialSum = cPolynomialSum + c;
            end

            qRoots = roots([cPolynomialSum(2), cPolynomialSum(3), cPolynomialSum(4), cPolynomialSum(1)]);
            qRoots = qRoots(imag(qRoots)==0);

            RzSolutions = cell(numel(qRoots),1);
            RSolutions  = cell(numel(qRoots),1);
            TSolutions  = cell(numel(qRoots),1);

            for qIndex = 1:numel(qRoots)

                qVal = qRoots(qIndex);

                RzCandidate = [ ...
                    (1/(qVal^2+1))*(-qVal^2+1), -2/(qVal^2+1)*qVal, 0; ...
                    2/(qVal^2+1)*qVal,          (1/(qVal^2+1))*(-qVal^2+1), 0; ...
                    0,                           0,                          1];

                RzSolutions{qIndex} = RzCandidate;
                RSolutions{qIndex} = RzCandidate * ...
                    imageBestPose(imageIndex).all_objects_onImg(1).Ry * ...
                    imageBestPose(imageIndex).all_objects_onImg(1).Rx;

                translationsThisRotation = [];

                for objIndex = 1:numel(imageBestPose(imageIndex).all_objects_onImg)

                    currentObject = imageBestPose(imageIndex).all_objects_onImg(objIndex);

                    Ellipsoid_T.a = currentObject.ellipsoidRadiusX;
                    Ellipsoid_T.b = currentObject.ellipsoidRadiusY;
                    Ellipsoid_T.c = currentObject.ellipsoidRadiusZ;
                    Ellipsoid_T.Rw_ell = currentObject.ellipsoidRotationMatrix;
                    Ellipsoid_T.Cw = currentObject.ellipsoidCenter;
                    Ellipsoid_T.Cell = [0;0;0];
                    Ellipsoid_T.Aell = diag([1/Ellipsoid_T.a^2, 1/Ellipsoid_T.b^2, 1/Ellipsoid_T.c^2]);

                    Cam_T.K = currentObject.cameraIntrinsicMatrix;
                    Cam_T.f = Cam_T.K(1,1);

                    estimatedRcam2world = RSolutions{qIndex};
                    R_cam_ell_exp = estimatedRcam2world' * Ellipsoid_T.Rw_ell;

                    Ellipse_T.Ri_el = currentObject.ellipseRotationMatrix;
                    Ellipse_T.a = currentObject.ellipseSemiMajorAxis;
                    Ellipse_T.b = currentObject.ellipseSemiMinorAxis;
                    Ellipse_T.Kc = (inv(Cam_T.K) * ...
                        [currentObject.ellipseCenter(1); currentObject.ellipseCenter(2); 1]) * Cam_T.f;

                    % "Perspective-1-Ellipsoid: Formulation, Analysis and Solutions of the
                    % Camera Pose Estimation Problem from One Ellipse-Ellipsoid Correspondence",
                    % from Vincent Gaudillière, Gilles Simon and Marie-Odile Berger.
                    %%
                    B_ = computeBackprojCone(Ellipse_T);
                    B_ell = R_cam_ell_exp' * B_ * R_cam_ell_exp;

                    [V, D] = eig(Ellipsoid_T.Aell, B_ell);
                    generalizedEigenvalues = sort(diag(D));

                    positiveEigenvalues = generalizedEigenvalues(generalizedEigenvalues > 0);
                    negativeEigenvalues = generalizedEigenvalues(generalizedEigenvalues < 0);

                    if numel(positiveEigenvalues) < numel(negativeEigenvalues)
                        multiplicity1Threshold = positiveEigenvalues;
                        multiplicity2Threshold = mean(negativeEigenvalues);
                    else
                        multiplicity2Threshold = mean(positiveEigenvalues);
                        multiplicity1Threshold = negativeEigenvalues;
                    end

                    [~, cols] = find(D == multiplicity1Threshold);
                    delta1 = V(:, cols);

                    planeNormal = [0 0 1];

                    k1 = sqrt(trace(inv(Ellipsoid_T.Aell)) - ...
                        (1/multiplicity2Threshold) * trace(inv(B_ell)));
                    k2 = -k1;

                    D_ell_1 = k1 * delta1;
                    D_ell_2 = k2 * delta1;

                    E_w_1 = Ellipsoid_T.Rw_ell * (D_ell_1 + Ellipsoid_T.Cell) + Ellipsoid_T.Cw;
                    E_w_2 = Ellipsoid_T.Rw_ell * (D_ell_2 + Ellipsoid_T.Cell) + Ellipsoid_T.Cw;

                    Rw_c = estimatedRcam2world;
                    Cc1 = Rw_c' * (Ellipsoid_T.Cw - E_w_1);
                    Cc2 = Rw_c' * (Ellipsoid_T.Cw - E_w_2);

                    Dd1_cam = [0;0;0] - Cc1;
                    Dd2_cam = [0;0;0] - Cc2;

                    if Cc1(3) > 0 && Cc2(3) < 0 && dot(Dd1_cam, planeNormal) < 0 && dot(Dd2_cam, planeNormal) > 0
                        E_w = E_w_1;
                    elseif Cc1(3) < 0 && Cc2(3) > 0 && dot(Dd2_cam, planeNormal) < 0 && dot(Dd1_cam, planeNormal) > 0
                        E_w = E_w_2;
                    else
                        error('Neither camera-center branch satisfies the chirality condition.');
                    end
                    %%
                    
                    translationsThisRotation = [translationsThisRotation; E_w']; 
                end

                TSolutions{qIndex} = translationsThisRotation;
            end

            imageBestPose(imageIndex).Rz_solutions = RzSolutions;
            imageBestPose(imageIndex).R_solutions = RSolutions;
            imageBestPose(imageIndex).T_solutions = TSolutions;
        end

        if numel(imageBestPose(imageIndex).inlier_objects) > 1

            bestMeanDirectionDifference = Inf;
            bestNumDirectionalInliers = -Inf;
            bestMeanIoUCurrentImage = -Inf;

            for rotationSolutionIndex = 1:numel(imageBestPose(imageIndex).R_solutions)

                RzCurrent = imageBestPose(imageIndex).Rz_solutions{rotationSolutionIndex};

                directionalErrorsCurrentRotation = [];
                numDirectionalInliers = 0;

                for objIndex = 1:numel(imageBestPose(imageIndex).inlier_objects)

                    currentObject = imageBestPose(imageIndex).inlier_objects(objIndex);

                    predDirX = currentObject.directionX;
                    predDirY = currentObject.directionY;
                    gtDirX   = currentObject.GTdirectionX;
                    gtDirY   = currentObject.GTdirectionY;
                    gtRz     = currentObject.Rz;

                    vc_in_world = RzCurrent * [predDirX; predDirY; 0];
                    vw_gt = gtRz * [gtDirX; gtDirY; 0];

                    vCross = cross(vc_in_world, vw_gt);
                    sinTheta = norm(vCross) / (norm(vc_in_world) * norm(vw_gt));
                    sinSqTheta = sinTheta^2;
                    vDot = dot(vc_in_world, vw_gt);

                    if sinSqTheta < sinSqThreshold && vDot > 0
                        directionalErrorsCurrentRotation = [directionalErrorsCurrentRotation, sinSqTheta]; 
                        numDirectionalInliers = numDirectionalInliers + 1;
                    end
                end

                meanDirectionDifference = mean(directionalErrorsCurrentRotation);

                if meanDirectionDifference < bestMeanDirectionDifference && ...
                        numDirectionalInliers >= bestNumDirectionalInliers

                    bestNumDirectionalInliers = numDirectionalInliers;
                    bestMeanDirectionDifference = meanDirectionDifference;

                    imageBestPose(imageIndex).best_Rz = imageBestPose(imageIndex).Rz_solutions{rotationSolutionIndex};
                    imageBestPose(imageIndex).best_R = imageBestPose(imageIndex).R_solutions{rotationSolutionIndex};

                    for translationSolutionIndex = 1:size(imageBestPose(imageIndex).T_solutions{rotationSolutionIndex,1}, 1)

                        TCurrent = imageBestPose(imageIndex).T_solutions{rotationSolutionIndex,1}(translationSolutionIndex,1:3)';
                        iouValuesCurrentRT = [];

                        for objIndex = 1:numel(imageBestPose(imageIndex).all_objects_onImg)

                            currentObject = imageBestPose(imageIndex).all_objects_onImg(objIndex);

                            currentCamera = createCamera( ...
                                TCurrent, ...
                                imageBestPose(imageIndex).best_R, ...
                                currentObject.cameraIntrinsicMatrix(1:3,1:3));

                            currentEllipsoid = createEllipsoid( ...
                                currentObject.ellipsoidCenter, ...
                                currentObject.ellipsoidRotationMatrix, ...
                                currentObject.ellipsoidRadiusX, ...
                                currentObject.ellipsoidRadiusY, ...
                                currentObject.ellipsoidRadiusZ);

                            projectedEllipse = projectEllipsoid(currentEllipsoid, currentCamera);

                            projectedEllipseVec = [ ...
                                projectedEllipse.center(1), ...
                                projectedEllipse.center(2), ...
                                projectedEllipse.ai, ...
                                projectedEllipse.bi, ...
                                atan2(projectedEllipse.Ri_el(2,1), projectedEllipse.Ri_el(1,1))];

                            detectedEllipseVec = [ ...
                                currentObject.ellipseCenter(1), ...
                                currentObject.ellipseCenter(2), ...
                                currentObject.ellipseSemiMajorAxis, ...
                                currentObject.ellipseSemiMinorAxis, ...
                                atan2(currentObject.ellipseRotationMatrix(2,1), ...
                                      currentObject.ellipseRotationMatrix(1,1))];

                            iouVal = calc_probiou(detectedEllipseVec, projectedEllipseVec);
                            iouValuesCurrentRT = [iouValuesCurrentRT, iouVal]; 
                        end

                        meanIoUCurrentRT = mean(iouValuesCurrentRT);

                        if meanIoUCurrentRT > bestMeanIoUCurrentImage
                            bestMeanIoUCurrentImage = meanIoUCurrentRT;
                            imageBestPose(imageIndex).best_mean_iou = meanIoUCurrentRT;
                            imageBestPose(imageIndex).best_T = TCurrent;
                        end
                    end

                    imageBestPose(imageIndex).err_rot = rad2deg(error_rotation( ...
                        imageBestPose(imageIndex).best_R, ...
                        imageBestPose(imageIndex).all_objects_onImg(1).GTRT(1:3,1:3)));

                    imageBestPose(imageIndex).err_trans = norm( ...
                        imageBestPose(imageIndex).best_T - ...
                        imageBestPose(imageIndex).all_objects_onImg(1).GTRT(1:3,4));
                end
            end

            perImageTimingSec(imageIndex) = toc(poseSolveStart);
        end

        % ========================================================
        % SUMMARY TABLE WITH IMAGE NAME + FINAL ERRORS
        % ========================================================

        imageNameForSummary = imageBestPose(imageIndex).image;
        numAllObjects = numel(imageBestPose(imageIndex).all_objects_onImg);
        numInlierObjects = numel(imageBestPose(imageIndex).inlier_objects);

        if isfield(imageBestPose(imageIndex), 'err_trans') && ~isempty(imageBestPose(imageIndex).err_trans)

            summaryRowCount = summaryRowCount + 1;
            summaryRows(summaryRowCount,:) = { ...
                char(imageNameForSummary), ...
                numAllObjects, ...
                numInlierObjects, ...
                imageBestPose(imageIndex).err_rot, ...
                imageBestPose(imageIndex).err_trans};
        end

        if isfield(imageBestPose(imageIndex), 'best_R') && isfield(imageBestPose(imageIndex), 'best_T')

            for objectIndex = 1:numel(imageBestPose(imageIndex).all_objects_onImg)

                currentObject = imageBestPose(imageIndex).all_objects_onImg(objectIndex);

                poseRecordCount = poseRecordCount + 1;

                poseRecordsRaw(poseRecordCount) = struct( ...
                    'imageName', currentObject.imageName, ...
                    'semanticId', currentObject.semanticId, ...
                    'instanceId', currentObject.instanceId, ...
                    'ellipsoidRadiusX', currentObject.ellipsoidRadiusX, ...
                    'ellipsoidRadiusY', currentObject.ellipsoidRadiusY, ...
                    'ellipsoidRadiusZ', currentObject.ellipsoidRadiusZ, ...
                    'ellipsoidRotationMatrix', currentObject.ellipsoidRotationMatrix, ...
                    'ellipsoidCenter', currentObject.ellipsoidCenter, ...
                    'cameraIntrinsicMatrix', currentObject.cameraIntrinsicMatrix, ...
                    'cameraCenterWorld', imageBestPose(imageIndex).best_T, ...
                    'estimatedRotationCamFromWorld', imageBestPose(imageIndex).best_R, ...
                    'translationErrorMeters', imageBestPose(imageIndex).err_trans, ...
                    'rotationErrorDegrees', imageBestPose(imageIndex).err_rot, ...
                    'ellipseRotationMatrix2D', currentObject.ellipseRotationMatrix, ...
                    'ellipseSemiMajorPixels', currentObject.ellipseSemiMajorAxis, ...
                    'ellipseSemiMinorPixels', currentObject.ellipseSemiMinorAxis, ...
                    'ellipseCenterPixels', currentObject.ellipseCenter, ...
                    'predictedDirectionX', currentObject.directionX, ...
                    'predictedDirectionY', currentObject.directionY, ...
                    'groundTruthDirectionX', currentObject.GTdirectionX, ...
                    'groundTruthDirectionY', currentObject.GTdirectionY, ...
                    'groundTruthPoseHomogeneous', currentObject.GTRT, ...
                    'rotationXCam', currentObject.Rx, ...
                    'rotationYCam', currentObject.Ry, ...
                    'rotationZCam', currentObject.Rz, ...
                    'rotationZEstimated', imageBestPose(imageIndex).best_Rz);
            end
        end
    end

    validTimingMask = ~ismissing(perImageTimingImageId) & strlength(perImageTimingImageId) > 0;
    timingTable = table(perImageTimingImageId(validTimingMask), perImageTimingSec(validTimingMask), ...
        'VariableNames', {'image_id','t_pose_s'});

    timingCsvPath = fullfile(config.resultsDir, sprintf('Timing_bestpose_seq%02d.csv', currentSequenceId));
    writetable(timingTable, timingCsvPath);
    fprintf('[OK] Sequence timing CSV saved: %s\n', timingCsvPath);
end

%% ============================================================
% EXPORT IMAGE-LEVEL SUMMARY TABLE
% Includes image name + final rotation/translation errors
%% ============================================================

if ~isempty(summaryRows)
    summaryTable = cell2table(summaryRows, ...
        'VariableNames', { ...
            'Image_Name', ...
            'All_Object_Count', ...
            'Inlier_Object_Count', ...
            'Final_Rotation_Error_Deg', ...
            'Final_Translation_Error'});

    summaryCsvPath = fullfile(config.resultsDir,  'VehBuildLSEsolver_image.csv');
    writetable(summaryTable, summaryCsvPath, 'Delimiter', ';');

    fprintf('[OK] Summary CSV saved: %s\n', summaryCsvPath);
end

%% ============================================================
% CONVERT TO poseRecords FORMAT
%% ============================================================

poseRecords = struct( ...
    'imageName', {}, ...
    'semanticId', {}, ...
    'instanceId', {}, ...
    'ellipsoidRadiusX', {}, ...
    'ellipsoidRadiusY', {}, ...
    'ellipsoidRadiusZ', {}, ...
    'ellipsoidRotationMatrix', {}, ...
    'ellipsoidCenter', {}, ...
    'cameraIntrinsicMatrix', {}, ...
    'translationMatrixT', {}, ...
    'rotationMatrixR', {}, ...
    'translationError', {}, ...
    'rotationError', {}, ...
    'ellipseRotationMatrix', {}, ...
    'ellipseSemiMajorAxis', {}, ...
    'ellipseSemiMinorAxis', {}, ...
    'ellipseCenter', {}, ...
    'directionX', {}, ...
    'directionY', {}, ...
    'GTdirectionX', {}, ...
    'GTdirectionY', {}, ...
    'GTRT', {}, ...
    'Rx', {}, ...
    'Ry', {}, ...
    'Rz', {}, ...
    'Rz_est', {} );

for recordIndex = 1:numel(poseRecordsRaw)

    poseRecords(recordIndex).imageName               = poseRecordsRaw(recordIndex).imageName;
    poseRecords(recordIndex).semanticId              = poseRecordsRaw(recordIndex).semanticId;
    poseRecords(recordIndex).instanceId              = poseRecordsRaw(recordIndex).instanceId;

    poseRecords(recordIndex).ellipsoidRadiusX        = poseRecordsRaw(recordIndex).ellipsoidRadiusX;
    poseRecords(recordIndex).ellipsoidRadiusY        = poseRecordsRaw(recordIndex).ellipsoidRadiusY;
    poseRecords(recordIndex).ellipsoidRadiusZ        = poseRecordsRaw(recordIndex).ellipsoidRadiusZ;

    poseRecords(recordIndex).ellipsoidRotationMatrix = poseRecordsRaw(recordIndex).ellipsoidRotationMatrix;
    poseRecords(recordIndex).ellipsoidCenter         = poseRecordsRaw(recordIndex).ellipsoidCenter;
    poseRecords(recordIndex).cameraIntrinsicMatrix   = poseRecordsRaw(recordIndex).cameraIntrinsicMatrix;

    poseRecords(recordIndex).translationMatrixT      = poseRecordsRaw(recordIndex).cameraCenterWorld;
    poseRecords(recordIndex).rotationMatrixR         = poseRecordsRaw(recordIndex).estimatedRotationCamFromWorld;
    poseRecords(recordIndex).translationError        = poseRecordsRaw(recordIndex).translationErrorMeters;
    poseRecords(recordIndex).rotationError           = poseRecordsRaw(recordIndex).rotationErrorDegrees;

    poseRecords(recordIndex).ellipseRotationMatrix   = poseRecordsRaw(recordIndex).ellipseRotationMatrix2D;
    poseRecords(recordIndex).ellipseSemiMajorAxis    = poseRecordsRaw(recordIndex).ellipseSemiMajorPixels;
    poseRecords(recordIndex).ellipseSemiMinorAxis    = poseRecordsRaw(recordIndex).ellipseSemiMinorPixels;
    poseRecords(recordIndex).ellipseCenter           = poseRecordsRaw(recordIndex).ellipseCenterPixels;

    poseRecords(recordIndex).directionX              = poseRecordsRaw(recordIndex).predictedDirectionX;
    poseRecords(recordIndex).directionY              = poseRecordsRaw(recordIndex).predictedDirectionY;
    poseRecords(recordIndex).GTdirectionX            = poseRecordsRaw(recordIndex).groundTruthDirectionX;
    poseRecords(recordIndex).GTdirectionY            = poseRecordsRaw(recordIndex).groundTruthDirectionY;

    poseRecords(recordIndex).GTRT                    = poseRecordsRaw(recordIndex).groundTruthPoseHomogeneous;

    poseRecords(recordIndex).Rx                      = poseRecordsRaw(recordIndex).rotationXCam;
    poseRecords(recordIndex).Ry                      = poseRecordsRaw(recordIndex).rotationYCam;
    poseRecords(recordIndex).Rz                      = poseRecordsRaw(recordIndex).rotationZCam;
    poseRecords(recordIndex).Rz_est                  = poseRecordsRaw(recordIndex).rotationZEstimated;
end

poseRecordsMatPath = fullfile(config.resultsDir, 'VehBuild_poseRecords_for_bestpose.mat');
save(poseRecordsMatPath, 'poseRecords', '-v7.3');

fprintf('[OK] Compatible poseRecords MAT saved: %s\n', poseRecordsMatPath);
fprintf('[INFO] Number of pose records: %d\n', numel(poseRecords));
fprintf('[INFO] Number of images with < 2 inliers: %d\n', numImagesWithLessThanTwoInliers);

disp('[DONE] Vehicle-build pose solver finished.');

end