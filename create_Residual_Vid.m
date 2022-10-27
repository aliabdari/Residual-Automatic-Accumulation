function create_Residual_Vid(Residual,name)

outputVideo = VideoWriter(name);
open(outputVideo);
img =mat2gray(Residual(:,:,:,1));
writeVideo(outputVideo,img)

for i = 2:size(Residual,4)
    img = mat2gray(Residual(:,:,:,i));
    writeVideo(outputVideo,img)
    clear img
end
size(Residual,4)
close(outputVideo);
end