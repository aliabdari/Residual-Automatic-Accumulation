function m=makematmovie(n)

if isempty(n)
    n = 5;
end

load lastmov

for i = 1:size(mov,4)
    %m(i).cdata = uint8([mov(:,:,:,i) mov2(:,:,:,i)]);
    m(i).cdata = uint8([mov2(:,:,:,i)]);
    m(i).colormap = [];
end


