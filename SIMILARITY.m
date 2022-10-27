function [score] = SIMILARITY(Y1, Y2)

A=abs(Y1);
B=abs(Y2);
T = 1;
quality_map = (2*A.*B + T) ./(A.^2+B.^2 + T); 
score = mean(mean(quality_map));

end