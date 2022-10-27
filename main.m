clear;
close;

addpath('MpegCode');

Quality_scaling=1;
glob_qinter=16;


path_first = 'E:\Darvish\UP-Fall-Dataset\UP-Fall-Dataset\';

addpath('mmread')

num_of_classes=dir(path_first);

for classes=3:numel(num_of_classes)
    className = strcat(num_of_classes(classes).name,'\')
    
    files1=strcat(path_first,className,'*.avi');
    number_of_videos=dir(files1);
    
    path_save1=strcat('.\Reconstructed_dataset\',className);
    mkdir(path_save1)
    
    path_save2=strcat('.\Reconstructed_dataset_accumulated\',className);
    mkdir(path_save2)
    
    for i=1:numel(number_of_videos)
        
        source1=strcat(path_first,className,number_of_videos(i).name);
        vid=mmread(source1);
        fprintf('vh=%d vw=%d nf=%d \n',vid.height,vid.width,vid.nrFramesTotal);
        no_frame = vid.nrFramesTotal;
        selectedFrames=selectFrames(vid.frames,224,224);
        fpat='IP';
        for j=1:size(selectedFrames,2)
            fpat=strcat(fpat,'P');
        end
        
        name=strrep(number_of_videos(i).name,'.mp4','.avi');
        mpeg=mpegproj(size(selectedFrames,2),fpat,selectedFrames,1,16,224,224,path_save1,path_save2,name);
    end
end




