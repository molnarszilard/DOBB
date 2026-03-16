import subprocess

scripts_dir = "./"
functionName = "pose_csv_to_poseRecords_minimal"
seq = 9

matlab_cmd = (
    f"addpath(genpath('{scripts_dir}'));"
    f"repoRoot=fileparts(which('{functionName}'));"
    f"dataRoot=fullfile(repoRoot,'Kitti360_VehicBuild');"
    f"resultsDir=fullfile(repoRoot,'results');"
    f"{functionName}(repoRoot,dataRoot,resultsDir,{seq});"
)

subprocess.run(["matlab", "-batch", matlab_cmd], check=True)

functionName = "abs_pose_lse_from_poseRecords"

matlab_cmd = (
    f"addpath(genpath('{scripts_dir}'));"
    f"repoRoot=fileparts(which('{functionName}'));"
    f"resultsDir=fullfile(repoRoot,'results');"
    f"iouMatFile=fullfile(resultsDir,'VehBuildMinimal_poseRecords_for_bestpose.mat');"
    f"{functionName}(repoRoot,resultsDir,iouMatFile);"
)

subprocess.run(["matlab", "-batch", matlab_cmd], check=True)