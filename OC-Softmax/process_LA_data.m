clear; close all; clc;

% This code is modified from the baseline system for ASVspoof 2019


addpath(genpath('LFCC'));
addpath(genpath('CQCC_v1.0'));

% set here the experiment to run (access and feature type)
access_type = 'LA';
feature_type = 'LFCC';

%'''
%|- path
%    |- path to VoiceWukong dataset
%        |- Alldataset
%        |- Alldataset32K
%        |- ...
%'''
pathToDatabase  = 'change to the path where Voicewukong is located';

pathToFeatures = horzcat('change to the path to save features',  '\ZH_Features\');


evalProtocolFile = fullfile(change this to the path to eval_list.txt, 'zh_eval_list.txt');



evalfileID = fopen(evalProtocolFile);
evalprotocol = textscan(evalfileID, '%s%s%s%s%s');
fclose(evalfileID);
evalfilelist = evalprotocol{2};




disp('Extracting features for evaluation data...');
for i=1:length(evalfilelist)

    
    filePath = fullfile(pathToDatabase,'VoiceWukong',[evalfilelist{i}]);
    
    [x,fs] = audioread(filePath);
    [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
    LFCC = [stat delta double_delta]';
    filename_LFCC = fullfile(pathToFeatures, horzcat('LFCC_', evalfilelist{i}, '.mat'))
    parsave(filename_LFCC, LFCC)
    LFCC = [];
end
disp('Done!');



function parsave(fname, x)
    [pathstr,~,~] = fileparts(fname);
    if ~exist(pathstr, 'dir')
       mkdir(pathstr);
    end
    save(fname, 'x') 
end
