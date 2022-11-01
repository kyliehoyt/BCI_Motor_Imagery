%To clal this function, numChEEG is the number EEG channels from the gdf
%file, numChBIP = 3, numChEOG = 0
function [eeg, eog, bip, header] = f2_load_EEG_BIP(gdf_filename,numChEEG,numChBIP,numChEOG)
    [tempSignal, header] = sload(gdf_filename);
    
    if size(tempSignal,2)>=64 && 0==2 % this was used to test a new configurations of electrodes
        channels = boolean([0,0,0,0,1,1,1,0,1,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,...
                            0,0,0,0,0,0,0,0,0,0,0,0,0,0]);
        eeg = tempSignal(:,channels);  
        header.Label = header.Label(channels);
    elseif size(tempSignal,2)>=64 % select the 32 channels only
        ch44 = ((contains(header.Label,'F') | contains(header.Label,'P') | contains(header.Label,'T') | contains(header.Label,'C')) & ~contains(header.Label,'AF') & ~contains(header.Label,'PO')& ~contains(header.Label,'FP'));
        % if you want to select 32 channels
        % channels=[1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    64    31    68];
        eeg = tempSignal(:,ch44); 
        eeg = eeg(:,1:numChEEG);
    elseif numChEEG == 15 && size(tempSignal,2)>numChEEG+numChBIP+1
        channels=[6,10,11,15,16,17,21,22,41,42,43,45,46,48,49];
        eeg = tempSignal(:,channels);
    else
        eeg = tempSignal(:,1:numChEEG);
    end
    if size(tempSignal,2) >= numChEEG+numChBIP
        bip = tempSignal(:,numChEEG+1:numChEEG+numChBIP);
        if size(bip,2)>1 && numChEOG>0
            eog = bip(:,1:numChEOG);
        else
            eog=[];
        end
    else
       eog = []; bip =[]; 
    end
end