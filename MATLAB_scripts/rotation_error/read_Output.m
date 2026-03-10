path_in = ['C:\Users\shiyi\Downloads\output.txt'];
p = [];
ss= 1;
lines  = readlines([path_in]);
for i = 1: size(lines,1)   
    tline = lines(i,:);
    if i < size(lines,1)-24
    alltline = lines(i:i+24,:);
    end
    if strcmp(tline{1},"")
        continue
    end
    if strcmp(tline{1}(1:5),'0000_')
        p(ss).img = {tline{1}(1:end-1)};
        kk = ss;
    end
    if strcmp(tline{1}(1:2),'Q1')
        partsQ1 = strsplit(tline{1}(5:end),' ');
        p(kk).Q1 = {partsQ1{1},partsQ1{2},partsQ1{3},partsQ1{4}};

        partsT1 = strsplit(alltline{2,1}(5:end),' ');
        p(kk).T1 = {partsT1{1},partsT1{2},partsT1{3}};

        parts1R1 = strsplit(alltline{3,1}(5:end),' ');
        parts2R1 = strsplit(alltline{4,1}(1:end),' ');
        parts3R1 = strsplit(alltline{5,1}(1:end),' ');
        p(kk).R1 = {parts1R1{1},parts1R1{2},parts1R1{3};parts2R1{1},parts2R1{2},parts2R1{3};parts3R1{1},parts3R1{2},parts3R1{3}};


        partsQ2 = strsplit(alltline{7,1}(5:end),' ');
        p(kk).Q2 = {partsQ2{1},partsQ2{2},partsQ2{3},partsQ2{4}};

        partsT2 = strsplit(alltline{8,1}(5:end),' ');
        p(kk).T2 = {partsT2{1},partsT2{2},partsT2{3}};

        parts1R2 = strsplit(alltline{9,1}(5:end),' ');
        parts2R2 = strsplit(alltline{10,1}(1:end),' ');
        parts3R2 = strsplit(alltline{11,1}(1:end),' ');
        p(kk).R2 = {parts1R2{1},parts1R2{2},parts1R2{3};parts2R2{1},parts2R2{2},parts2R2{3};parts3R2{1},parts3R2{2},parts3R2{3}};
       
        partsQ3 = strsplit(alltline{13,1}(5:end),' ');
        p(kk).Q3 = {partsQ3{1},partsQ3{2},partsQ3{3},partsQ3{4}};

        partsT3 = strsplit(alltline{14,1}(5:end),' ');
        p(kk).T3 = {partsT3{1},partsT3{2},partsT3{3}};

        parts1R3 = strsplit(alltline{14,1}(5:end),' ');
        parts2R3 = strsplit(alltline{15,1}(1:end),' ');
        parts3R3 = strsplit(alltline{17,1}(1:end),' ');
        p(kk).R3 = {parts1R3{1},parts1R3{2},parts1R3{3};parts2R3{1},parts2R3{2},parts2R3{3};parts3R3{1},parts3R3{2},parts3R3{3}};

        partsQ4 = strsplit(alltline{19,1}(5:end),' ');
        p(kk).Q4 = {partsQ4{1},partsQ4{2},partsQ4{3},partsQ4{4}};

        partsT4 = strsplit(alltline{20,1}(5:end),' ');
        p(kk).T4 = {partsT4{1},partsT4{2},partsT4{3}};

        parts1R4 = strsplit(alltline{21,1}(5:end),' ');
        parts2R4 = strsplit(alltline{22,1}(1:end),' ');
        parts3R4 = strsplit(alltline{23,1}(1:end),' ');
        p(kk).R4 = {parts1R4{1},parts1R4{2},parts1R4{3};parts2R4{1},parts2R4{2},parts2R4{3};parts3R4{1},parts3R4{2},parts3R4{3}};

    end
    

    ss = ss+1;

end

del = [];
for j = 1: size(p,2)
    if isempty(p(j).Q1)
        del = [del,j];
    end
end

p(:,del) = [];
pose = p;
save(['C:\Users\shiyi\Downloads\main\main\pose.mat'],'pose');
