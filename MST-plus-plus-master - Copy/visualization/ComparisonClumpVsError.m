clc; clear all; close all;

% User input for how many times the program should run
num_runs = 1;

% User input for selecting image (901-950)
user_input = input('Enter image number (901-950): ', 's');

% Define base folders - copy data path
exp_base_folder = 'C:\Users\Michael\OneDrive\Desktop\Research\Optical Material Characterization\MST-plus-plus-master\test_develop_code\exp';
dataset_folder = 'C:\Users\Michael\OneDrive\Desktop\Research\Optical Material Characterization\MST-plus-plus-master\dataset\Train_Spec';
rgb_folder = 'C:\Users\Michael\OneDrive\Desktop\Research\Optical Material Characterization\MST-plus-plus-master\dataset\Train_RGB';

% Get list of all subfolders in the experimental directory
subfolders = dir(exp_base_folder);
subfolders = subfolders([subfolders.isdir]); % Keep only directories
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'})); % Remove '.' and '..'

% Construct dataset file path
dataset_file = fullfile(dataset_folder, sprintf('ARAD_1K_%04d.mat', str2double(user_input)));

% Check if dataset file exists
if ~exist(dataset_file, 'file')
    error('Dataset file not found: %s', dataset_file);
end

% Load ground truth image
dataset_data = load(dataset_file);
fields = fieldnames(dataset_data);

% Try to extract the hyperspectral image
dataset_image = [];
for i = 1:length(fields)
    temp_data = dataset_data.(fields{i});
    if ndims(temp_data) == 3 % Ensure it's a 3D matrix
        dataset_image = temp_data;
        break;
    end
end

% If no valid hyperspectral image was found, print the issue and exit
if isempty(dataset_image)
    error('No valid hyperspectral image found in %s. Check the file contents.', dataset_file);
end

% Load corresponding ground truth RGB image
gt_file = fullfile(rgb_folder, sprintf('ARAD_1K_0%03d.jpg', str2double(user_input)));
if exist(gt_file, 'file')
    ground_truth_img = imread(gt_file);
    fprintf('Loaded ground truth: %s\n', gt_file);
else
    error('Ground truth image not found: %s', gt_file);
end

% Initialize a structure to hold data from all subfolders
exp_data_all = struct();

% Loop through each subfolder and try to load the file
for i = 1:length(subfolders)
    subfolder_name = subfolders(i).name;
    exp_file = fullfile(exp_base_folder, subfolder_name, sprintf('ARAD_1K_%04d.mat', str2double(user_input)));

    if exist(exp_file, 'file')
        exp_data = load(exp_file);
        exp_fields = fieldnames(exp_data);

        % Extract hyperspectral image from the loaded file
        for j = 1:length(exp_fields)
            temp_data = exp_data.(exp_fields{j});
            if ndims(temp_data) == 3 % Ensure it's 3D
                exp_data_all.(subfolder_name) = temp_data;
                fprintf('Loaded: %s\n', exp_file);
                break;
            end
        end
    else
        fprintf('File not found in %s: %s\n', subfolder_name, exp_file);
    end
end

% Define wavelengths (400nm to 700nm, 31 bands)
wavelengths = 400:10:700; % Wavelengths from 400nm to 700nm, 31 bands

% Initialize arrays to store MRAE and RMSE values for each clump size
clump = 40;
clump_sizes = 1:clump;  % Example clump sizes from 1 to 10
MRAE_values_all = zeros(num_runs, length(clump_sizes));
RMSE_values_all = zeros(num_runs, length(clump_sizes));

% Select a random pixel for each iteration
[x_dim, y_dim, ~] = size(dataset_image);
rand_x = randi(x_dim);
rand_y = randi(y_dim);

% For testing, using a fixed pixel
% rand_x = 244;
% rand_y = 390;

% Loop through the number of runs
for clump_idx = 1:length(clump_sizes)
    clump_size = clump_sizes(clump_idx);  % Set current clump size
    
    for run_idx = 1:num_runs
        % Calculate the clump boundaries (ensure the clump is within image bounds)
        x_start = max(rand_x - floor(clump_size / 2), 1);
        x_end = min(rand_x + floor(clump_size / 2), x_dim);
        y_start = max(rand_y - floor(clump_size / 2), 1);
        y_end = min(rand_y + floor(clump_size / 2), y_dim);

        % Extract the clump of pixels and average the values
        clump_data = dataset_image(x_start:x_end, y_start:y_end, :);
        avg_spectrum = mean(clump_data, [1, 2]);  % Average over the spatial dimensions (x, y)

        % Initialize metrics
        MRAE_values = zeros(1, length(subfolders));
        RMSE_values = zeros(1, length(subfolders));

        % Loop over each method (subfolder)
        flds = fieldnames(exp_data_all);
        for i = 1:length(flds)
            exp_image = exp_data_all.(flds{i});
            exp_clump_data = exp_image(x_start:x_end, y_start:y_end, :);
            avg_exp_spectrum = mean(exp_clump_data, [1, 2]);  % Average over the spatial dimensions (x, y)

            % Ensure both spectra are row vectors for comparison
            dataset_spectrum = avg_spectrum(:)';
            avg_exp_spectrum = avg_exp_spectrum(:)';

            % Calculate MRAE
            MRAE_values(i) = mean(abs(dataset_spectrum - avg_exp_spectrum) ./ dataset_spectrum);

            % Calculate RMSE
            RMSE_values(i) = sqrt(mean((dataset_spectrum - avg_exp_spectrum).^2));
        end

        % Store the MRAE and RMSE values for the current run and clump size
        MRAE_values_all(run_idx, clump_idx) = mean(MRAE_values);
        RMSE_values_all(run_idx, clump_idx) = mean(RMSE_values);
    end
end

% Plot Ground Truth Image in a new window
figure;
imshow(ground_truth_img);
hold on;
plot(rand_y, rand_x, 'gs', 'MarkerSize', clump, 'LineWidth', 2); % Hollow green marker
title('Ground Truth Image');

% Plot MRAE vs Clump Size in a separate window
figure;
subplot(2,1,1)
plot(clump_sizes, mean(MRAE_values_all, 1), '-o', 'LineWidth', 2);
title('MRAE vs Clump Size');
xlabel('Clump Size');
ylabel('MRAE');
grid on;

% Plot RMSE vs Clump Size in another separate window
subplot(2,1,2)
plot(clump_sizes, mean(RMSE_values_all, 1), '-o', 'LineWidth', 2);
title('RMSE vs Clump Size');
xlabel('Clump Size');
ylabel('RMSE');
grid on;
