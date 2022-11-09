addpath('C:\Program Files\MATLAB\R2022b\toolbox\')
electrode = "Dry";
subject = 4;
session = 1;
session_type = "Offline";
folder = "Subject_00" + string(subject) + "_Session_00" + string(session) + "_" + session_type + "_Visual_" + electrode;
folderpath = pwd + "\" + folder;
addpath(folderpath);
fileList = dir(fullfile(folder, '*.gdf'));
runnum = 1;
for f = 1:length(fileList)
    [s, h] = sload(fileList(f).name);
    sname = "s" + string(runnum);
    hname = "h" + string(runnum);
    save(sname, "s");
    save(hname, "h");
    runnum = runnum+1;
end
