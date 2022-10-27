function create_Accumulated_Vid(Residual,name,window_size)
size(Residual,4)

outputVideo = VideoWriter(name);
open(outputVideo)

img = mat2gray(Residual(:,:,:,1));
writeVideo(outputVideo,img)

ii=1;

window_accumulation=zeros(1,window_size);

img=zeros(224,224,3);

for ii = 2:window_size+1
    img = img + mat2gray(Residual(:,:,:,ii));
    %     writeVideo(outputVideo,img)
    window_accumulation(ii-1) = SIMILARITY(Residual(:,:,1,ii),Residual(:,:,1,ii+1));
    % writeVideo(outputVideo,abs(img)/65536);
end

img=img./max(max(max(img)));
writeVideo(outputVideo,img)

criteria = mean(window_accumulation);
fprintf('criteria=%f \n',criteria)
img=zeros(224,224,3);
ii=window_size+2;
counter_each_accumulation=0;
num_new_video_frames=0;

while ii<size(Residual,4)
    %             ii
    
    current_similarity = SIMILARITY(mat2gray(Residual(:,:,1,ii)),mat2gray(Residual(:,:,1,ii-1)));
    
    if(current_similarity>criteria || counter_each_accumulation==0)
        m2g = mat2gray(Residual(:,:,:,ii));
        
        img = img + m2g;
        
        ii=ii+1;
        counter_each_accumulation = counter_each_accumulation + 1;
        %         img=mat2gray(img);
        
        window_accumulation(1,1:end-1)=window_accumulation(1,2:end);
        window_accumulation(1,end)= current_similarity;
        criteria = mean(window_accumulation);
        fprintf('criteria=%f \n',criteria)
    else
        %                 ii=ii-1;
        window_accumulation=zeros(1,window_size);
        num_new_video_frames = num_new_video_frames+1;
        counter_each_accumulation = 0;
        img=img./max(max(max(img)));
        writeVideo(outputVideo,img);
        img=zeros(224,224,3);
        index = 1;
        for jj = ii:ii+window_size-1
            if jj < size(Residual,4)
                img = img + mat2gray(Residual(:,:,:,jj));
                %                 writeVideo(outputVideo,img)
                window_accumulation(index) = SIMILARITY(Residual(:,:,1,jj),Residual(:,:,1,jj+1));
                index = index + 1;
            else
                img=img./max(max(max(img)));
                writeVideo(outputVideo,img)
                img=zeros(224,224,3);
                break
            end
            img=img./max(max(max(img)));
            writeVideo(outputVideo,img)
            img=zeros(224,224,3);
        end
        ii = ii + window_size + 1;
    end
    
    
end

close(outputVideo);
size(Residual,4)
end