clear;
close all;
clc;

subfolderPath = 'C:\Users\au752405\OneDrive - Aarhus universitet\Desktop\test';

% Get the list of files in the subfolder
files = dir(fullfile(subfolderPath, '*.csv'));

% Check if the subfolder exists
if isempty(files)
    error('No .csv files found in the specified subfolder.');
end

% Extract paths of .csv files
csvFilePaths = cell(size(files));
for i = 1:length(files)
    csvFilePaths{i} = fullfile(subfolderPath, files(i).name);
end

% Extract names of .csv files
csvFileNames = cell(size(files));
for i = 1:length(files)
    [~, csvFileNames{i}, ~] = fileparts(files(i).name);
end

% Create position .png from every acceleration
for i=1:length(csvFilePaths)
    path = csvFilePaths{i};
    disp(path)
    [t, accx, accy, accz] = csv2acc(path);
    [pos, vel] = acc2Imag(t,accx,accy,accz);
    
    plot(pos(:,1),pos(:,2), Color='black');
    xlim([-0.6 0.6]);
    ylim([-0.6 0.6]);
    axis off;
    
    % Create subfolder if it doesn't exist
    folderName = 'Test_Dataset_img\5';
    if ~exist(folderName, 'dir')
        mkdir(folderName);
    end
    
    % Save the plot as a PNG image in the subfolder
    imageName = fullfile(folderName, plus(csvFileNames{i},".png"));

    fig = gcf;
    %fig.Position(3:4) = [256, 256]; % [width, height]
    % Export the plot to a PNG file
    %set(fig, 'PaperUnits', 'points');
    %x_width=256;
    %y_width=256;
    %set(fig, 'PaperPosition', [0 0 x_width y_width]); %
    saveas(fig,imageName)
    %print(fig,imageName,'Resolution',300)
   
    disp(['Image saved as ' imageName]);
end







