function [target_matrix,modify_Data]=modify_dataset_zero_class(Data)
target=Data(:,end);
[n1,n2]=size(target);
[m1,m2]=max(target);
a2=find(target==0);
if isempty(a2)==false 
target_matrix=zeros(n1,m1+1);
for i=1:n1
    a=target(i,:);
    for j=1:m1+1
    if j-1==a
    target_matrix(i,j)=1; 
    end
    end
end
%
else
 target_matrix=zeros(n1,m1); 
 for i=1:n1
    a=target(i,:);
    for j=1:m1
    if j==a
    target_matrix(i,j)=1; 
    end
    end
%}
end

end

modify_Data=[Data(:,1:end-1) target_matrix];