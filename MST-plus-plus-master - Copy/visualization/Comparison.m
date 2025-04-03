clc; clear all; close all;

% User input for how many times the program should run
num_runs = input('Enter the number of times to run the program: ');

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

% Create a more distinct color palette (using unique, well-spaced colors)
custom_colors = [
    1.0000, 0.0000, 0.0000;  % Red
    0.0000, 1.0000, 0.0000;  % Green
    0.0000, 0.0000, 1.0000;  % Blue
    1.0000, 1.0000, 0.0000;  % Yellow
    1.0000, 0.0000, 1.0000;  % Magenta
    0.0000, 1.0000, 1.0000;  % Cyan
    0.5000, 0.0000, 0.5000;  % Purple
    1.0000, 0.7500, 0.8000;  % Pink
    0.5000, 0.5000, 0.5000;  % Gray
    0.5600, 0.9300, 0.5600;  % Light Green
];

% Loop through the number of runs
for run_idx = 1:num_runs
    fprintf('\nRunning iteration %d of %d...\n', run_idx, num_runs);
    
    % Select a random pixel for each iteration
    [x_dim, y_dim, ~] = size(dataset_image);
    rand_x = randi(x_dim);
    rand_y = randi(y_dim);

    % Create a figure with two subplots: RGB image and spectral comparison
    figure('WindowState', 'maximized'); % Open figure in full screen

    % Plot the RGB image in the first subplot
    subplot(1, 2, 1);
    imshow(ground_truth_img);
    hold on;
    % Highlight the selected pixel on the RGB image (hollow circle)
    plot(rand_y, rand_x, 'go', 'MarkerFaceColor', 'none', 'MarkerSize', 10, 'LineWidth', 2); % Hollow green circle
    title(['Ground Truth RGB Image - ARAD 1K 0', user_input, ' - Iteration ', num2str(run_idx)]);

    % Plot the spectral comparison in the second subplot
    subplot(1, 2, 2); hold on;

    % Ground truth spectrum
    dataset_spectrum = squeeze(dataset_image(rand_x, rand_y, :));
    
    % Initialize metrics
    MRAE_values = zeros(1, length(subfolders));
    RMSE_values = zeros(1, length(subfolders));
    
    % Loop over each method (subfolder)
    flds = fieldnames(exp_data_all);
    for i = 1:length(flds)
        exp_image = exp_data_all.(flds{i});
        exp_spectrum = squeeze(exp_image(rand_x, rand_y, :));
        
        % Calculate MRAE
        MRAE_values(i) = mean(abs(dataset_spectrum - exp_spectrum) ./ dataset_spectrum);
        
        % Calculate RMSE
        RMSE_values(i) = sqrt(mean((dataset_spectrum - exp_spectrum).^2));
        
        % Plot the spectrum of the current method
        plot(wavelengths, exp_spectrum, 'Color', custom_colors(mod(i-1, size(custom_colors, 1)) + 1, :), 'LineWidth', 1.5, 'DisplayName', flds{i});
    end

    % Plot ground truth spectrum
    plot(wavelengths, dataset_spectrum, '-k', 'LineWidth', 2, 'DisplayName', 'Ground Truth');

    legend;
    title(sprintf('Spectral Comparison at Pixel (%d, %d) - Iteration %d', rand_x, rand_y, run_idx));
    xlabel('Wavelength (nm)'); ylabel('Intensity');
    grid on; hold off;
    
    % Display MRAE and RMSE for each method
    fprintf('MRAE and RMSE for pixel (%d, %d):\n', rand_x, rand_y);
    for i = 1:length(flds)
        fprintf('%s - MRAE: %.4f, RMSE: %.4f\n', flds{i}, MRAE_values(i), RMSE_values(i));
    end
end
