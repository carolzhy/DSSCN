function [trial]=normal_class(x)
[m,n]=size(x);
for i=1:n
   trial(:,i)=abs(x(:,i))/abs(max(x(:,i)));
    
end
%tes2=abs(x1)/abs(max(x1));
%trial1=data1/abs(max(data1));
%tes3=tes1/abs(max(tes1));