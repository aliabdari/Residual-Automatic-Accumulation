function selectedFrames=selectFrames(frames,vh,vw)

index=1;
for j=1:1:size(frames,2)
	s.cdata=imresize(frames(j).cdata,[vh,vw]);
        s.colormap=[];
    selectedFrames(index)=s;
    index=index+1;
    if(index > 60000)
        break
    end
end
end 