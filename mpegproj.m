%%
function mpeg=mpegproj(no_frame,fpat,mov,Quality_scaling,glob_qinter,height,width,path_save1,path_save2,name)
% Example of MPEG-2 style coding. This function will get a movie, encode
% it, decode it, and write the MPEG data and both movie versions to disk
% for later playback and analysis. (See companion functions 'playlast' and
% 'quiverplot'.)
%
% By Steve Hoelzer
% 2005-4-18
%
% ECE 434, Multimedia Communication Networks
% Professor Marilyn Andrews
% University of Illinois at Chicago (UIC)

fprintf('\nMPEG Project\n')
% global mpeg;
% global no_frame;
nf = no_frame; % Number of frames to process, 0 = process entire movie

fprintf('%i frames\n',nf)

mov = getmov(nf,mov);

tic
mpeg = encmov(mov,fpat,Quality_scaling,glob_qinter,path_save1,path_save2,name,height,width);
% save(name,'mpeg');
fprintf('Encode time: %s\n',sec2timestr(toc))

% editted
% tic
% mov2 = decmpeg(mpeg);
% fprintf('Decode time: %s\n',sec2timestr(toc))

% save lastmov mov mpeg mov2


%%
function movdata = getmov(nf,mov)
% Choose the movie to process. Each movie must be saved to disk in a .mat
% file. The .mat file should contain a variable named 'mov' that is a
% MATLAB movie.
%
% See loadFileYuv and loadFileY4m for help converting movies from YUV and
% Y4M format to MATLAB movies.

% Movie files & number of frames
%
% bus 150
% carphone 382
% tempete2 260
% husky 250
% akiyo 300 @ 60 fps
% foreman 300
% hall_monitor 300 @ 60 fps
% mobile 300 @ 60 fps

% load foreman_qcif.yuv
% load bus
% load carphone
% load tempete2
% load mobile
% load hall_monitor
% load husky


% global mov;


% Movie is RGB, so get 4-D array (height, width, color component, frame #)
if nf == 0
    nf = length(mov);
end
movdata = repmat(uint8(0),[size(mov(1).cdata), nf]);
for i = 1:nf
    movdata(:,:,:,i) = mov(i).cdata;
end


%%
function mpeg = encmov(mov,fpat,Quality_scaling,glob_qinter,path_save1,path_save2,name,height,width)

% Frame type pattern (repeats for entire movie)
% global fpat; % Custom
% fpat = 'I'; % All I frames
% fpat = ['I', repmat('P',1,size(mov,4))]; % All P frames (except #1)

% Loop over frames
k = 0;
pf = [];
%editted
% progressbar
Residual=zeros(height,width,3,size(mov,4));



for i = 1:size(mov,4)
    fprintf('progress = %f no_frames = %d \n',i/size(mov,4),size(mov,4));
    % Get frame
    f = double(mov(:,:,:,i));
    
    % Convert frame to YCrCb
    f = rgb2ycc(f);
    
    % Get frame type
    k = k + 1;
    if k > length(fpat)
        k = 1;
    end
    ftype = fpat(k);
    
    % Encode frame
    [mpeg{i},pf,res] = encframe(f,ftype,pf,Quality_scaling,glob_qinter,height,width);%i
    Residual(:,:,1,i)=res(:,:,1);
    Residual(:,:,2,i)=res(:,:,2);
    Residual(:,:,3,i)=res(:,:,3);
    
    
end



createVid(Residual,fullfile(path_save1,name));
acc_similarity_acc_window_members_every_intruption(Residual,fullfile(path_save2,name),5);

clear Residual
% close(outputVideo);


%%
function [mpeg,df,Residual] = encframe(f,ftype,pf,Quality_scaling,glob_qinter,height,width)

[M,N,i] = size(f);
mbsize = [M, N] / 16;
% mbsize = [M, N] / 8;
mpeg = struct('type',[],'mvx',[],'mvy',[],'scale',[],'coef',[]);
mpeg(mbsize(1),mbsize(2)).type = [];

% Loop over macroblocks
pfy = pf(:,:,1);
df = zeros(size(f));
Residual=zeros(height,width,3);
for m = 1:mbsize(1)
    for n = 1:mbsize(2)
        
        % Encode one macroblock
        x = 16*(m-1)+1 : 16*(m-1)+16;
        y = 16*(n-1)+1 : 16*(n-1)+16;
        %         x = 8*(m-1)+1 : 8*(m-1)+8;
        %         y = 8*(n-1)+1 : 8*(n-1)+8;
        [mpeg(m,n),df(x,y,:),res] = encmacroblock(f(x,y,:),ftype,pf,pfy,x,y,Quality_scaling,glob_qinter);
        Residual(x,y,1)=res(:,:,1);
        Residual(x,y,2)=res(:,:,2);
        Residual(x,y,3)=res(:,:,3);
    end % macroblock loop
end % macroblock loop


%%
function b = getblocks(mb)

b = zeros([8, 8, 6]);
% b = zeros([4, 4, 6]);
% Four lum blocks
b(:,:,1) = mb( 1:8,  1:8,  1);
b(:,:,2) = mb( 1:8,  9:16, 1);
b(:,:,3) = mb( 9:16, 1:8,  1);
b(:,:,4) = mb( 9:16, 9:16, 1);
% b(:,:,1) = mb( 1:4,  1:4,  1);
% b(:,:,2) = mb( 1:4,  5:8, 1);
% b(:,:,3) = mb( 5:8, 1:4,  1);
% b(:,:,4) = mb( 5:8, 5:8, 1);

% Two subsampled chrom blocks (mean of four neighbors)
b(:,:,5) = 0.25 * ( mb(1:2:15,1:2:15, 2) + mb(1:2:15,2:2:16, 2) ...
    + mb(2:2:16,1:2:15, 2) + mb(2:2:16,2:2:16, 2) );
b(:,:,6) = 0.25 * ( mb(1:2:15,1:2:15, 3) + mb(1:2:15,2:2:16, 3) ...
    + mb(2:2:16,1:2:15, 3) + mb(2:2:16,2:2:16, 3) );
% b(:,:,5) = 0.25 * ( mb(1:2:7,1:2:7, 2) + mb(1:2:7,2:2:8, 2) ...
%                   + mb(2:2:8,1:2:7, 2) + mb(2:2:8,2:2:8, 2) );
% b(:,:,6) = 0.25 * ( mb(1:2:7,1:2:7, 3) + mb(1:2:7,2:2:8, 3) ...
%                   + mb(2:2:8,1:2:7, 3) + mb(2:2:8,2:2:8, 3) );


%%
function ycc = rgb2ycc(rgb)

% Transformation matrix
m = [ 0.299     0.587     0.144;
    -0.168736 -0.331264  0.5;
    0.5      -0.418688 -0.081312];

% Get movie data
[nr,nc,c] = size(rgb);

% Reshape for matrix multiply
rgb = reshape(rgb,nr*nc,3);

% Transform color coding
ycc = m * rgb';
ycc = ycc + repmat([0; 0.5; 0.5],1,nr*nc);

% Reshape to original size
ycc = reshape(ycc',nr,nc,3);


%%
function [mpeg,dmb,res] = encmacroblock(mb,ftype,pf,pfy,x,y,Quality_scaling,glob_qinter)

% Coeff quantization matrices
persistent q1 q2
if isempty(q1)
    q1 = qintra;
    q2 = qinter(glob_qinter);
end

% Quality scaling
% global Quality_scaling;
scale = Quality_scaling;

% Init mpeg struct
mpeg.type = 'I';
mpeg.mvx = 0;
mpeg.mvy = 0;

% res=zeros(8,8,3);
% Find motion vectors
if ftype == 'P'
    mpeg.type = 'P';
    [mpeg,emb] = getmotionvec(mpeg,mb,pf,pfy,x,y);
    mb = emb; % Set macroblock to error for encoding
    %     res=emb(:,:,1);
    % res=emb;
    q = q2;
else
    q = q1;
end

% Get lum and chrom blocks
b = getblocks(mb);

% Encode blocks
for i = 6:-1:1
    mpeg.scale(i) = scale;
    coef = dct2(b(:,:,i));
    mpeg.coef(:,:,i) = round( 8 * coef ./ (scale * q) );
end

% Decode this macroblock for reference by a future P frame
[dmb,res] = decmacroblock(mpeg,pf,x,y,glob_qinter);


%%
function [mpeg,emb] = getmotionvec(mpeg,mb,pf,pfy,x,y)

% Do search in Y only
mby = mb(:,:,1);
[M,N] = size(pfy);

% Exhaustive search
maxstep = 10; % Largest allowable motion vector in x and y

mvxv = -maxstep:maxstep;
mvyv = -maxstep:maxstep;
minsad = inf;
for i = 1:length(mvxv)
    
    tx = x + mvxv(i);
    if (tx(1) < 1) | (M < tx(end))
        continue
    end
    
    for j = 1:length(mvyv)
        
        ty = y + mvyv(j);
        if (ty(1) < 1) | (N < ty(end))
            continue
        end
        
        sad = sum(sum(abs(mby-pfy(tx,ty))));
        
        if sad < minsad
            minsad = sad;
            mvx = mvxv(i);
            mvy = mvyv(j);
        end
        
    end
end

% % Logarithmic search
% step = 8; % Initial step size for logarithmic search
%
% dx = [0 1 1 0 -1 -1 -1  0  1]; % Unit direction vectors
% dy = [0 0 1 1  1  0 -1 -1 -1]; % [origin, right, right-up, up, left-up,
%                                %         left, left-down, down, right-down]
%
% mvx = 0;
% mvy = 0;
% while step >= 1
%
%     minsad = inf;
%     for i = 1:length(dx)
%
%         tx = x + mvx + dx(i)*step;
%         if (tx(1) < 1) || (M < tx(end))
%             continue
%         end
%
%         ty = y + mvy + dy(i)*step;
%         if (ty(1) < 1) || (N < ty(end))
%             continue
%         end
%
%         sad = sum(sum(abs(mby-pfy(tx,ty))));
%
%         if sad < minsad
%             ii = i;
%             minsad = sad;
%         end
%
%     end
%
%     mvx = mvx + dx(ii)*step;
%     mvy = mvy + dy(ii)*step;
%
%     step = step / 2;
%
% end

mpeg.mvx = mvx; % Store motion vectors
mpeg.mvy = mvy;

emb = mb - pf(x+mvx,y+mvy,:); % Error macroblock


%%
function mov = decmpeg(mpeg)

movsize = size(mpeg{1});
mov = repmat(uint8(0),[16*movsize(1:2), 3, length(mpeg)]);

% Loop over frames
pf = [];
for i = 1:length(mpeg)
    
    % Decode frame
    f = decframe(mpeg{i},pf);
    
    % Cache previous frame
    pf = f;
    
    % Convert frame to RGB
    f = ycc2rgb(f);
    
    % Make sure movie is in 8 bit range
    f = min( max(f,0), 255);
    
    % Store frame
    mov(:,:,:,i) = uint8(f);
    %editted
    %     progressbar((i+length(mpeg))/(2*length(mpeg)))
    
end


%%
function fr = decframe(mpeg,pf)

mbsize = size(mpeg);
M = 16 * mbsize(1);
N = 16 * mbsize(2);

% Loop over macroblocks
fr = zeros(M,N,3);
for m = 1:mbsize(1)
    for n = 1:mbsize(2)
        
        % Construct frame
        x = 16*(m-1)+1 : 16*(m-1)+16;
        y = 16*(n-1)+1 : 16*(n-1)+16;
        [fr(x,y,:),zzz] = decmacroblock(mpeg(m,n),pf,x,y);
        
    end % macroblock loop
end % macroblock loop


%%
function mb = putblocks(b)

% mb = zeros([16, 16, 3]);
mb = zeros([8, 8, 3]);

% Four lum blocks
mb( 1:8,  1:8,  1) = b(:,:,1);
mb( 1:8,  9:16, 1) = b(:,:,2);
mb( 9:16, 1:8,  1) = b(:,:,3);
mb( 9:16, 9:16, 1) = b(:,:,4);

% mb( 1:4,  1:4,  1) = b(:,:,1);
% mb( 1:4,  5:8, 1) = b(:,:,2);
% mb( 5:8, 1:4,  1) = b(:,:,3);
% mb( 5:8, 5:8, 1) = b(:,:,4);

% Two subsampled chrom blocks
z = [1 1; 1 1];
mb(:,:,2) = kron(b(:,:,5),z);
mb(:,:,3) = kron(b(:,:,6),z);


%%
function rgb = ycc2rgb(ycc)

% Transformation matrix
m = [ 0.299     0.587     0.144;
    -0.168736 -0.331264  0.5;
    0.5      -0.418688 -0.081312];
m = inv(m);

% Get movie data
[nr,nc,c] = size(ycc);

% Reshape for matrix multiply
ycc = reshape(ycc,nr*nc,3);

% Transform color coding
rgb = ycc - repmat([0, 0.5, 0.5],nr*nc,1);
rgb = m * rgb';

% Reshape to original size
rgb = reshape(rgb',nr,nc,3);


%%
function [mb,res] = decmacroblock(mpeg,pf,x,y,glob_qinter)

% Coeff quantization matrices
persistent q1 q2
if isempty(q1)
    q1 = qintra;
    q2 = qinter(glob_qinter);
end

mb = zeros(16,16,3);
% mb = zeros(8,8,3);

% Predict with motion vectors
if mpeg.type == 'P'
    mb = pf(x+mpeg.mvx,y+mpeg.mvy,:);
    q = q2;
else
    q = q1;
end

% Decode blocks
for i = 6:-1:1
    coef = mpeg.coef(:,:,i) .* (mpeg.scale(i) * q) / 8;
    b(:,:,i) = idct2(coef);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct macroblock
res=putblocks(b);
mb = mb + res;



%%
function q = qinter(glob_qinter)
% Quantization table for P or B frames
% global glob_qinter;
% q = repmat(16,8,8);
q = glob_qinter;


%%
function q = qintra
% Quantization table for I frames

q = [ 8 16 19 22 26 27 29 34;
    16 16 22 24 27 29 34 37;
    19 22 26 27 29 34 34 38;
    22 22 26 27 29 34 37 40;
    22 26 27 29 32 35 40 48;
    26 27 29 32 35 40 48 58;
    26 27 29 34 38 46 56 69;
    27 29 35 38 46 56 69 83 ];

% q = [ 8 16 19 22 ;
%      16 16 22 24 ;
%      19 22 26 27 ;
%      22 22 26 27 ;];



%%
function y = dct2(x)
% Perform 2-D DCT

% Use dctmtx to compute IDCT faster
persistent d
if isempty(d)
    d = dctmtx(8);
    % d = dctmtx(4);
end

y = d * x * d';

% % DCT is seperable so compute on columns, then on rows
% y = dct(x); % Columns
% y = dct(y')'; % Rows

%%
function y = idct2(x)
% Perform 2-D IDCT

% Use dctmtx to compute IDCT faster
persistent d
if isempty(d)
    d = dctmtx(8);
    %       d = dctmtx(4);
end

y = d' * x * d;

% % DCT is seperable so compute on columns, then on rows
% y = idct(x); % Columns
% y = idct(y')'; % Rows
