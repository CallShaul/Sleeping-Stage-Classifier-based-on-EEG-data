clear all;
%% Load mat file
files = ["02", "05", "07", "08","10", "15", "18", "20", "23", "25"];
folder = 'C:\Users\saul6\Documents\Electrooptical Eng\Brain waves signal processing\Asignment\10studies\Original data\';
for i=1:length(files)

    num = num2str(files(i));
    temp = load([folder,'26', num , '.mat'],['Sig26', num],['Hyp26', num]);
    eval(['temp_data = temp.Sig26', num, ';']);
    eval(['GT = temp.Hyp26', num, ';']);
    clear temp;

    hz200 = 1;
    hz10 = 1;
    hz1 = 1;
    
    for j=1:length(temp_data)
        
        if cell2mat(temp_data(j,2)) == 200
            
            data_200hz(:,hz200) = cell2mat(temp_data(j,3));
            
            hz200 = hz200 + 1;
        end
        
        if cell2mat(temp_data(j,2)) == 10
            
            data_10hz(:,hz10) = cell2mat(temp_data(j,3));
            
            hz10 = hz10 + 1;
        end
        
        if cell2mat(temp_data(j,2)) == 1
            
            data_1hz(:,hz1) = cell2mat(temp_data(j,3));
            
            hz1 = hz1 + 1;
        end
        
    end
    
    data_200hz_titles = {'F3A2';'F4A1';'C3A2';'C4A1';'O1A2';'O2A1';'LOC';'ROC';'A1A2';'Chin';'RLeg';'LLeg';'ECG';'Snore';'PPG';'imp'};
    data_10hz_titles = {'rr';'PFlo';'TFlo';'CFlo';'Tho';'Abd';'SpO2';'Leak'};
    data_1hz_titles = {'EPAP';'IPAP';'Body'};
    
    save(['26', num, '_py'], 'GT', 'data_200hz', 'data_10hz', 'data_1hz', 'data_200hz_titles', 'data_10hz_titles', 'data_1hz_titles')
    
    clear data_200hz data_10hz data_1hz data_200hz_titles data_10hz_titles data_1hz_titles GT;
end
