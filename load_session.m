%addpath('C:\Program Files\MATLAB\R2022b\toolbox\')

% electrode = "Dry";
% subject = 4;
% session = 1;
% session_type = "Offline";
% folder = "Subject_00" + string(subject) + "_Session_00" + string(session) + "_" + session_type + "_Visual_" + electrode;
% folderpath = pwd + "\" + folder;
% addpath(folderpath);
% fileList = dir(fullfile(folder, '*.gdf'));
% runnum = 1;
% for f = 1:length(fileList)
%     [s, h] = sload(fileList(f).name);
%     sname = "s" + string(runnum);
%     hname = "h" + string(runnum);
%     save(sname, "s");
%     save(hname, "h");
%     runnum = runnum+1;
% end

electrodes = ["Dry", "Gel"];
session_types = ["Online", "Offline"];


for subject = 4:6
    for electrode_id = 1:length(electrodes)
        for session_type_id = 1:length(session_types)

            electrode = electrodes{electrode_id}
            session_type = session_types{session_type_id}
            
            % Save session_1
            session = 1;
            folder = "raw_data/Subject_00" + string(subject) + "_Session_00" + string(session) + "_" + session_type + "_Visual_" + electrode;
            folderpath = pwd + "/" + folder;
            addpath(folderpath);

            save_path = "subject_" + string(subject) + "/" + electrode + "/" + session_type + "/session_" + string(session);
            save_path = lower(save_path);
            mkdir(pwd + "/" + save_path);
            fileList = dir(fullfile(folder, '*.gdf'));
            addpath(folderpath);
            runnum = 1;
            for f = 1:length(fileList)
                [s, h] = sload(fileList(f).name);
                sname = save_path + "/s" + string(runnum);
                hname = save_path + "/h" + string(runnum);
                save(sname, "s");
                save(hname, "h");
                runnum = runnum+1;
            end

            if(session_type == "Online")
                % Save session 2
                session = 2;
                folder = "raw_data/Subject_00" + string(subject) + "_Session_00" + string(session) + "_" + session_type + "_Visual_" + electrode;
                folderpath = pwd + "/" + folder;
                addpath(folderpath);
    
                save_path = "subject_" + string(subject) + "/" + electrode + "/" + session_type + "/session_" + string(session);
                save_path = lower(save_path);
                mkdir(pwd + "/" + save_path);
                fileList = dir(fullfile(folder, '*.gdf'));
                addpath(folderpath);
                runnum = 1;
                for f = 1:length(fileList)
                    [s, h] = sload(fileList(f).name);
                    sname = save_path + "/s" + string(runnum);
                    hname = save_path + "/h" + string(runnum);
                    save(sname, "s");
                    save(hname, "h");
                    runnum = runnum+1;
                end
            end
        end
    end

end


% # File Structure
% # > subject_<id>
% #     > gel
% #        > offline
% #            > session_1
% #                > h1.mat
% #                > s1.mat
% #        > online
% #            > session_1
% #                > h1.mat
% #                > s1.mat
% #            > session_2
% #                > h1.mat
% #                > s1.mat
% #     > dry
% #        > offline
% #            > session_1
% #                > h1.mat
% #                > s1.mat
% #        > online
% #            > session_1
% #                > h1.mat
% #                > s1.mat
% #            > session_2
% #                > h1.mat
% #                > s1.mat