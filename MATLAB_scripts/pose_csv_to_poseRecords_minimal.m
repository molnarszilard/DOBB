function pose_csv_to_poseRecords_minimal(repoRoot, dataRoot, resultsDir, sequences)

config = struct();
config.repoRoot   = repoRoot;
config.dataRoot   = dataRoot;
config.resultsDir = resultsDir;

if ~exist(config.resultsDir,'dir'), mkdir(config.resultsDir); end

config.imageSubdir = "image_00";
config.sequences  = sequences(:)';



%% ===== CONFIG =====
% config = struct();
% config.repoRoot   = fileparts(mfilename('fullpath'));
% config.dataRoot   = fullfile(config.repoRoot, "Kitti360_VehicBuild");
% config.resultsDir = fullfile(config.repoRoot, "results");
% if ~exist(config.resultsDir,'dir'), mkdir(config.resultsDir); end
% 
% config.imageSubdir = "image_00";
% config.sequences = [ 9 ];  

addpath(genpath(fullfile(config.repoRoot, "Functions")));
addpath(genpath(fullfile(config.repoRoot, "rotation_error")));

%% ===== PREALLOC =====
globalMatchedObjectCount      = 0;
totalObjectCount              = 0;
allPredictedRowIndices        = [];
matchedPredictedRowIndices    = [];

totalPredictedRowCount        = 0;
processedObjectCount          = 0;

allRotationErrorsDeg          = [];
allTranslationErrorsM         = [];

dataImageNames                = strings(0,1);
dataSemanticIds               = [];
dataInstanceIds               = [];
predictedImageNames           = strings(0,1);
predictedSemanticIds          = [];
predictedInstanceIds          = [];

minRotationErrorPerImage      = [];
maxRotationErrorPerImage      = [];
minTranslationErrorPerImage   = [];
maxTranslationErrorPerImage   = [];
rotationAtMinTranslation      = [];
processedCsvFileNames         = strings(0,1);

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

%% ===== MAIN LOOP OVER SEQUENCES =====
for sequenceLoopIndex = 1:numel(config.sequences)
    sequenceId = config.sequences(sequenceLoopIndex);
    fprintf('[%d/%d] Sequence %02d\n', sequenceLoopIndex, numel(config.sequences), sequenceId);

    sequenceImageDir = fullfile( ...
        config.dataRoot, ...
        sprintf('2023_05_28_drive_%04d_sync', sequenceId), ...
        config.imageSubdir);

    dataCsvFiles = dir(fullfile(sequenceImageDir, 'ellipse_data', '*.csv'));
    gtCsvFiles   = dir(fullfile(sequenceImageDir, 'ellipse_dir_data_gt', '*.csv'));
    predCsvFiles = dir(fullfile(sequenceImageDir, 'ellipse_dir_data_pred', '*.csv'));

    [isGtCommon, ~]   = ismember({gtCsvFiles.name},   {dataCsvFiles.name});
    [isPredCommon, ~] = ismember({predCsvFiles.name}, {dataCsvFiles.name});
    gtCsvFiles   = gtCsvFiles(isGtCommon);
    predCsvFiles = predCsvFiles(isPredCommon);

    dataCsvNames = sort({dataCsvFiles.name});
    gtCsvNames   = sort({gtCsvFiles.name});
    predCsvNames = sort({predCsvFiles.name});

    if ~isequal(dataCsvNames, gtCsvNames, predCsvNames)
        error('Missing CSV files across data / gt / pred folders.');
    end

    numberOfImages = numel(dataCsvNames);

    imageIdsThisSequence = strings(numberOfImages, 1);
    poseSolveTimePerImageSec = nan(numberOfImages, 1);

    sequencePrefix = sprintf('%02d_', sequenceId);

    for imageIndex = 1:numberOfImages
        csvFileName = dataCsvNames{imageIndex};
        imageBaseName = string(erase(csvFileName, '.csv'));
        imageId = string(sequencePrefix) + imageBaseName;
        imageIdsThisSequence(imageIndex) = imageId;

        currentImageCsvName = dataCsvNames{imageIndex};

        dataCsvPath = fullfile(sequenceImageDir, 'ellipse_data', dataCsvNames{imageIndex});
        gtCsvPath   = fullfile(sequenceImageDir, 'ellipse_dir_data_gt', gtCsvNames{imageIndex});
        predCsvPath = fullfile(sequenceImageDir, 'ellipse_dir_data_pred', predCsvNames{imageIndex});

        dataTable = readtable(dataCsvPath, 'Delimiter', ';', 'PreserveVariableNames', true);
        gtTable   = readtable(gtCsvPath,   'Delimiter', ';', 'PreserveVariableNames', true);
        predTable = readtable(predCsvPath, 'Delimiter', ';', 'PreserveVariableNames', true);

        totalPredictedRowCount = totalPredictedRowCount + size(predTable, 1);

        rotationErrorsThisImage = [];
        translationErrorsThisImage = [];
        totalPoseTimeThisImageSec = 0;

        for objectRowIndex = 1:size(dataTable, 1)
            totalObjectCount = totalObjectCount + 1;

            if isempty(predTable) || ~ismember(dataTable.instanceId(objectRowIndex), predTable.instanceId)
                continue;
            end

            matchingPredictedRows = find( ...
                predTable.instanceId == dataTable.instanceId(objectRowIndex) & ...
                predTable.semanticId == dataTable.semanticId(objectRowIndex));

            allPredictedRowIndices = [allPredictedRowIndices; matchingPredictedRows(:)];

            if isempty(matchingPredictedRows)
                continue;
            end

            globalMatchedObjectCount = globalMatchedObjectCount + 1;
            matchedPredictedRowIndices = [matchedPredictedRowIndices; matchingPredictedRows(:)];
            predictedRowIndex = matchingPredictedRows(1);

            rotX = predTable.rotX(predictedRowIndex);
            rotY = predTable.rotY(predictedRowIndex);
            rotZ = gtTable.rotZ(objectRowIndex);

            rotationXCam = [ ...
                1 0 0;
                0 cos(rotX) -sin(rotX);
                0 sin(rotX)  cos(rotX)];

            rotationYCam = [ ...
                 cos(rotY) 0 sin(rotY);
                 0         1 0;
                -sin(rotY) 0 cos(rotY)];

            rotationZCam = [ ...
                cos(rotZ) -sin(rotZ) 0;
                sin(rotZ)  cos(rotZ) 0;
                0          0         1];

            groundTruthRotationCamFromWorld = rotationZCam * rotationYCam * rotationXCam;

            predictedDirectionX = predTable.directionX(predictedRowIndex);
            predictedDirectionY = predTable.directionY(predictedRowIndex);
            predictedThetaRad   = predTable.theta(predictedRowIndex);

            predictedDirectionAngleRad = atan2(predictedDirectionY, predictedDirectionX);
            estimatedYawZRad = predictedThetaRad - predictedDirectionAngleRad;

            rotationZEstimated = [ ...
                cos(estimatedYawZRad) -sin(estimatedYawZRad) 0;
                sin(estimatedYawZRad)  cos(estimatedYawZRad) 0;
                0                      0                     1];

            poseStartTime = tic;

            estimatedRotationCamFromWorld = rotationZEstimated * rotationYCam * rotationXCam;

            dataImageNames      = [dataImageNames; char(string(sequencePrefix) + string(dataTable.ImageName{objectRowIndex}))]; 
            dataSemanticIds     = [dataSemanticIds; dataTable.semanticId(objectRowIndex)];
            dataInstanceIds     = [dataInstanceIds; dataTable.instanceId(objectRowIndex)];
            predictedImageNames = [predictedImageNames; string(predTable.ImageName(predictedRowIndex))];
            predictedSemanticIds = [predictedSemanticIds; predTable.semanticId(predictedRowIndex)];
            predictedInstanceIds = [predictedInstanceIds; predTable.instanceId(predictedRowIndex)];

            rotationErrorDeg = rad2deg(error_rotation(estimatedRotationCamFromWorld, groundTruthRotationCamFromWorld));
            if ~isreal(rotationErrorDeg)
                rotationErrorDeg = real(rotationErrorDeg);
            end
            allRotationErrorsDeg = [allRotationErrorsDeg; rotationErrorDeg];

            ellipsoid = struct();
            ellipsoid.a = dataTable.ellipsoidRadiusX(objectRowIndex);
            ellipsoid.b = dataTable.ellipsoidRadiusY(objectRowIndex);
            ellipsoid.c = dataTable.ellipsoidRadiusZ(objectRowIndex);
            ellipsoid.rotationWorldFromEllipsoid = [ ...
                dataTable.ellipsoidR_ell2w_11(objectRowIndex), dataTable.ellipsoidR_ell2w_12(objectRowIndex), dataTable.ellipsoidR_ell2w_13(objectRowIndex);
                dataTable.ellipsoidR_ell2w_21(objectRowIndex), dataTable.ellipsoidR_ell2w_22(objectRowIndex), dataTable.ellipsoidR_ell2w_23(objectRowIndex);
                dataTable.ellipsoidR_ell2w_31(objectRowIndex), dataTable.ellipsoidR_ell2w_32(objectRowIndex), dataTable.ellipsoidR_ell2w_33(objectRowIndex)];
            ellipsoid.centerWorld = [ ...
                dataTable.ellipsoidCenterX(objectRowIndex);
                dataTable.ellipsoidCenterY(objectRowIndex);
                dataTable.ellipsoidCenterZ(objectRowIndex)];
            ellipsoid.centerLocal = [0; 0; 0];
            ellipsoid.quadricMatrix = diag([1/ellipsoid.a^2, 1/ellipsoid.b^2, 1/ellipsoid.c^2]);

            camera = struct();
            camera.K = [ ...
                dataTable.cameraIntrinsic_11(objectRowIndex), dataTable.cameraIntrinsic_12(objectRowIndex), dataTable.cameraIntrinsic_13(objectRowIndex);
                dataTable.cameraIntrinsic_21(objectRowIndex), dataTable.cameraIntrinsic_22(objectRowIndex), dataTable.cameraIntrinsic_23(objectRowIndex);
                dataTable.cameraIntrinsic_31(objectRowIndex), dataTable.cameraIntrinsic_32(objectRowIndex), dataTable.cameraIntrinsic_33(objectRowIndex)];
            camera.focalLength = dataTable.cameraIntrinsic_11(objectRowIndex);
            camera.centerWorld = [ ...
                dataTable.cameraTranslationX(objectRowIndex);
                dataTable.cameraTranslationY(objectRowIndex);
                dataTable.cameraTranslationZ(objectRowIndex)];

            rotationCamFromEllipsoid = estimatedRotationCamFromWorld' * ellipsoid.rotationWorldFromEllipsoid;

            ellipseMeasurement = struct();
            ellipseMeasurement.rotation2D = [ ...
                predTable.ellipseRotationMatrixR11(predictedRowIndex), predTable.ellipseRotationMatrixR12(predictedRowIndex);
                predTable.ellipseRotationMatrixR21(predictedRowIndex), predTable.ellipseRotationMatrixR22(predictedRowIndex)];
            ellipseMeasurement.semiMajorPixels = predTable.("a_axis(Semi-Major px)")(predictedRowIndex);
            ellipseMeasurement.semiMinorPixels = predTable.("b_axis(Semi-Minor px)")(predictedRowIndex);
            ellipseMeasurement.centerRay = (inv(camera.K) * [ ...
                predTable.("ellipseCenterX(px)")(predictedRowIndex);
                predTable.("ellipseCenterY(px)")(predictedRowIndex);
                1]) * camera.focalLength;

            ellipseCone = struct();
            ellipseCone.Ri_el = ellipseMeasurement.rotation2D;
            ellipseCone.a     = ellipseMeasurement.semiMajorPixels;
            ellipseCone.b     = ellipseMeasurement.semiMinorPixels;
            ellipseCone.Kc    = ellipseMeasurement.centerRay;
            
            % "Perspective-1-Ellipsoid: Formulation, Analysis and Solutions of the
            % Camera Pose Estimation Problem from One Ellipse-Ellipsoid Correspondence",
            % from Vincent Gaudillière, Gilles Simon and Marie-Odile Berger.
            %%
            backprojectionCone = computeBackprojCone(ellipseCone);
            coneInEllipsoidFrame = rotationCamFromEllipsoid' * backprojectionCone * rotationCamFromEllipsoid;

            [eigenVectors, eigenValues] = eig(ellipsoid.quadricMatrix, coneInEllipsoidFrame);
            lambdaValues = real(diag(eigenValues));
            [sortedLambdaValues, sortedIndices] = sort(lambdaValues);

            gap12 = abs(sortedLambdaValues(2) - sortedLambdaValues(1));
            gap23 = abs(sortedLambdaValues(3) - sortedLambdaValues(2));

            if gap12 > gap23
                singleEigenvalueIndex = sortedIndices(1);
                repeatedEigenvalueIndices = sortedIndices(2:3);
            else
                singleEigenvalueIndex = sortedIndices(3);
                repeatedEigenvalueIndices = sortedIndices(1:2);
            end

            deltaVector = eigenVectors(:, singleEigenvalueIndex);
            deltaVector = deltaVector / norm(deltaVector);

            repeatedEigenvalueMean = mean(lambdaValues(repeatedEigenvalueIndices));
            kSquared = trace(inv(ellipsoid.quadricMatrix)) - ...
                       (1 / repeatedEigenvalueMean) * trace(inv(coneInEllipsoidFrame));
            kValue = sqrt(max(kSquared, 0));

            cameraCenterLocalCandidate1 =  kValue * deltaVector;
            cameraCenterLocalCandidate2 = -kValue * deltaVector;

            cameraCenterWorldCandidate1 = ellipsoid.rotationWorldFromEllipsoid * cameraCenterLocalCandidate1 + ellipsoid.centerWorld;
            cameraCenterWorldCandidate2 = ellipsoid.rotationWorldFromEllipsoid * cameraCenterLocalCandidate2 + ellipsoid.centerWorld;

            rotationWorldFromCamEstimated = estimatedRotationCamFromWorld.';
            ellipsoidCenterInCamCandidate1 = rotationWorldFromCamEstimated * (ellipsoid.centerWorld - cameraCenterWorldCandidate1);
            ellipsoidCenterInCamCandidate2 = rotationWorldFromCamEstimated * (ellipsoid.centerWorld - cameraCenterWorldCandidate2);

            isCandidate1Valid = (ellipsoidCenterInCamCandidate1(3) > 0);
            isCandidate2Valid = (ellipsoidCenterInCamCandidate2(3) > 0);

            if isCandidate1Valid && ~isCandidate2Valid
                estimatedCameraCenterWorld = cameraCenterWorldCandidate1;
            elseif isCandidate2Valid && ~isCandidate1Valid
                estimatedCameraCenterWorld = cameraCenterWorldCandidate2;
            else
                error('No unique camera-center solution in front of the camera.');
            end
            %%
            
            translationErrorMeters = norm(estimatedCameraCenterWorld - camera.centerWorld);
            allTranslationErrorsM = [allTranslationErrorsM; translationErrorMeters];

            processedObjectCount = processedObjectCount + 1;
            rotationErrorsThisImage = [rotationErrorsThisImage; rotationErrorDeg];
            translationErrorsThisImage = [translationErrorsThisImage; translationErrorMeters];

            totalPoseTimeThisImageSec = totalPoseTimeThisImageSec + toc(poseStartTime);

            if ~isequal(dataTable.instanceId(objectRowIndex), predTable.instanceId(predictedRowIndex))
                error('Instance ID mismatch.');
            end

            poseRecordCount = poseRecordCount + 1;
            poseRecords(poseRecordCount) = struct( ...
                'imageName', char(string(sequencePrefix) + string(dataTable.ImageName{objectRowIndex})), ...
                'semanticId', predTable.semanticId(predictedRowIndex), ...
                'instanceId', predTable.instanceId(predictedRowIndex), ...
                'ellipsoidRadiusX', dataTable.ellipsoidRadiusX(objectRowIndex), ...
                'ellipsoidRadiusY', dataTable.ellipsoidRadiusY(objectRowIndex), ...
                'ellipsoidRadiusZ', dataTable.ellipsoidRadiusZ(objectRowIndex), ...
                'ellipsoidRotationMatrix', [ ...
                    dataTable.ellipsoidR_ell2w_11(objectRowIndex), dataTable.ellipsoidR_ell2w_12(objectRowIndex), dataTable.ellipsoidR_ell2w_13(objectRowIndex);
                    dataTable.ellipsoidR_ell2w_21(objectRowIndex), dataTable.ellipsoidR_ell2w_22(objectRowIndex), dataTable.ellipsoidR_ell2w_23(objectRowIndex);
                    dataTable.ellipsoidR_ell2w_31(objectRowIndex), dataTable.ellipsoidR_ell2w_32(objectRowIndex), dataTable.ellipsoidR_ell2w_33(objectRowIndex)], ...
                'ellipsoidCenter', [ ...
                    dataTable.ellipsoidCenterX(objectRowIndex);
                    dataTable.ellipsoidCenterY(objectRowIndex);
                    dataTable.ellipsoidCenterZ(objectRowIndex)], ...
                'cameraIntrinsicMatrix', camera.K, ...
                'cameraCenterWorld', estimatedCameraCenterWorld, ...
                'estimatedRotationCamFromWorld', estimatedRotationCamFromWorld, ...
                'translationErrorMeters', translationErrorMeters, ...
                'rotationErrorDegrees', rotationErrorDeg, ...
                'ellipseRotationMatrix2D', [ ...
                    predTable.ellipseRotationMatrixR11(predictedRowIndex), predTable.ellipseRotationMatrixR12(predictedRowIndex);
                    predTable.ellipseRotationMatrixR21(predictedRowIndex), predTable.ellipseRotationMatrixR22(predictedRowIndex)], ...
                'ellipseSemiMajorPixels', ellipseMeasurement.semiMajorPixels, ...
                'ellipseSemiMinorPixels', ellipseMeasurement.semiMinorPixels, ...
                'ellipseCenterPixels', [ ...
                    predTable.("ellipseCenterX(px)")(predictedRowIndex);
                    predTable.("ellipseCenterY(px)")(predictedRowIndex)], ...
                'predictedDirectionX', predTable.directionX(predictedRowIndex), ...
                'predictedDirectionY', predTable.directionY(predictedRowIndex), ...
                'groundTruthDirectionX', gtTable.directionX(objectRowIndex), ...
                'groundTruthDirectionY', gtTable.directionY(objectRowIndex), ...
                'groundTruthPoseHomogeneous', [groundTruthRotationCamFromWorld, camera.centerWorld; 0 0 0 1], ...
                'rotationXCam', rotationXCam, ...
                'rotationYCam', rotationYCam, ...
                'rotationZCam', rotationZCam, ...
                'rotationZEstimated', rotationZEstimated);
        end

        poseSolveTimePerImageSec(imageIndex) = totalPoseTimeThisImageSec;

        if ~isempty(rotationErrorsThisImage)
            minRotationError = min(rotationErrorsThisImage);
            maxRotationError = max(rotationErrorsThisImage);
            [minTranslationError, minTranslationIndex] = min(translationErrorsThisImage);
            maxTranslationError = max(translationErrorsThisImage);

            minRotationErrorPerImage    = [minRotationErrorPerImage; minRotationError];
            maxRotationErrorPerImage    = [maxRotationErrorPerImage; maxRotationError];
            minTranslationErrorPerImage = [minTranslationErrorPerImage; minTranslationError];
            maxTranslationErrorPerImage = [maxTranslationErrorPerImage; maxTranslationError];
            rotationAtMinTranslation    = [rotationAtMinTranslation; rotationErrorsThisImage(minTranslationIndex)];
            processedCsvFileNames       = [processedCsvFileNames; string(currentImageCsvName)];
        end
    end

    %% ===== WRITE PER IMAGE TIMING =====
    validImageIds = imageIdsThisSequence(:);
    validPoseTimes = poseSolveTimePerImageSec(:);

    validMask = ~ismissing(validImageIds) & strlength(validImageIds) > 0 & isfinite(validPoseTimes);
    validImageIds = validImageIds(validMask);
    validPoseTimes = validPoseTimes(validMask);

    [uniqueImageIds, ~, groupIndices] = unique(validImageIds, 'stable');
    summedPoseTimes = accumarray(groupIndices, validPoseTimes, [], @sum);

    timingPerImageTable = table(uniqueImageIds, summedPoseTimes, ...
        'VariableNames', {'image_id', 't_pose_s'});

    timingCsvPath = fullfile(config.resultsDir, sprintf('Timing_VehBuildMinimal_total%02d.csv', sequenceId));
    writetable(timingPerImageTable, timingCsvPath);
    fprintf('[OK] Per-image summed timing -> %s (rows=%d)\n', timingCsvPath, height(timingPerImageTable));
end

%% ===== FINAL =====
dataImageNames      = string(dataImageNames(:));
dataSemanticIds     = dataSemanticIds(:);
dataInstanceIds     = dataInstanceIds(:);
predictedImageNames = string(predictedImageNames(:));
predictedSemanticIds = predictedSemanticIds(:);
predictedInstanceIds = predictedInstanceIds(:);

allRotationErrorsDeg = allRotationErrorsDeg(:);
allTranslationErrorsM = allTranslationErrorsM(:);

perObjectResultsTable = table( ...
    dataImageNames, dataSemanticIds, dataInstanceIds, ...
    predictedImageNames, predictedSemanticIds, predictedInstanceIds, ...
    allRotationErrorsDeg, allTranslationErrorsM, ...
    'VariableNames', { ...
        'dataImageName', ...
        'dataSemanticId', ...
        'dataInstanceId', ...
        'predImageName', ...
        'predSemanticId', ...
        'predInstanceId', ...
        'rotationErrorDegGTvsPred', ...
        'translationErrorMetersDetEll'});

perObjectCsvPath = fullfile(config.resultsDir, 'VehBuildMinimal_solution_abs_per_object.csv');
writetable(perObjectResultsTable, perObjectCsvPath, 'Delimiter', ';');
fprintf('[OK] Per-object metrics -> %s\n', perObjectCsvPath);

%% ===== CONVERT poseRecords TO SECOND-SCRIPT FORMAT =====
rawPoseRecords = poseRecords;

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

for recordIndex = 1:numel(rawPoseRecords)
    poseRecords(recordIndex).imageName               = rawPoseRecords(recordIndex).imageName;
    poseRecords(recordIndex).semanticId              = rawPoseRecords(recordIndex).semanticId;
    poseRecords(recordIndex).instanceId              = rawPoseRecords(recordIndex).instanceId;

    poseRecords(recordIndex).ellipsoidRadiusX        = rawPoseRecords(recordIndex).ellipsoidRadiusX;
    poseRecords(recordIndex).ellipsoidRadiusY        = rawPoseRecords(recordIndex).ellipsoidRadiusY;
    poseRecords(recordIndex).ellipsoidRadiusZ        = rawPoseRecords(recordIndex).ellipsoidRadiusZ;

    poseRecords(recordIndex).ellipsoidRotationMatrix = rawPoseRecords(recordIndex).ellipsoidRotationMatrix;
    poseRecords(recordIndex).ellipsoidCenter         = rawPoseRecords(recordIndex).ellipsoidCenter;
    poseRecords(recordIndex).cameraIntrinsicMatrix   = rawPoseRecords(recordIndex).cameraIntrinsicMatrix;

    poseRecords(recordIndex).translationMatrixT      = rawPoseRecords(recordIndex).cameraCenterWorld;
    poseRecords(recordIndex).rotationMatrixR         = rawPoseRecords(recordIndex).estimatedRotationCamFromWorld;
    poseRecords(recordIndex).translationError        = rawPoseRecords(recordIndex).translationErrorMeters;
    poseRecords(recordIndex).rotationError           = rawPoseRecords(recordIndex).rotationErrorDegrees;

    poseRecords(recordIndex).ellipseRotationMatrix   = rawPoseRecords(recordIndex).ellipseRotationMatrix2D;
    poseRecords(recordIndex).ellipseSemiMajorAxis    = rawPoseRecords(recordIndex).ellipseSemiMajorPixels;
    poseRecords(recordIndex).ellipseSemiMinorAxis    = rawPoseRecords(recordIndex).ellipseSemiMinorPixels;
    poseRecords(recordIndex).ellipseCenter           = rawPoseRecords(recordIndex).ellipseCenterPixels;

    poseRecords(recordIndex).directionX              = rawPoseRecords(recordIndex).predictedDirectionX;
    poseRecords(recordIndex).directionY              = rawPoseRecords(recordIndex).predictedDirectionY;
    poseRecords(recordIndex).GTdirectionX            = rawPoseRecords(recordIndex).groundTruthDirectionX;
    poseRecords(recordIndex).GTdirectionY            = rawPoseRecords(recordIndex).groundTruthDirectionY;

    poseRecords(recordIndex).GTRT                    = rawPoseRecords(recordIndex).groundTruthPoseHomogeneous;

    poseRecords(recordIndex).Rx                      = rawPoseRecords(recordIndex).rotationXCam;
    poseRecords(recordIndex).Ry                      = rawPoseRecords(recordIndex).rotationYCam;
    poseRecords(recordIndex).Rz                      = rawPoseRecords(recordIndex).rotationZCam;
    poseRecords(recordIndex).Rz_est                  = rawPoseRecords(recordIndex).rotationZEstimated;
end

poseRecordsMatPath = fullfile(config.resultsDir, 'VehBuildMinimal_poseRecords_for_bestpose.mat');
save(poseRecordsMatPath, 'poseRecords', '-v7.3');
fprintf('[OK] Compatible poseRecords MAT saved -> %s (rows=%d)\n', poseRecordsMatPath, numel(poseRecords));

end

