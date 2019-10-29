function [y,residual]=inferencefull(z,Center_upper,Center_lower,Spread,design_factor,pik,model,delta)
[N,nm]=size(z);
[R,n]=size(Center_upper);
m=nm-n;
residual=zeros(N,m);
y=zeros(N,m);
for k=1:N
 di_upper=zeros(R,n);
    di_lower=zeros(R,n);
    di1_lower=zeros(R,1);
    di1_upper=zeros(R,1);
    lambda=zeros(R,m);
    z_current=z(k,:);
  for i=1:R-1
                  dist=(z_current(1:n)-Center_lower(i,:))*Spread(:,:,i)*(z_current(1:n)-Center_lower(i,:))';
            dist1=(z_current(1:n)-Center_upper(i,:))*Spread(:,:,i)*(z_current(1:n)-Center_upper(i,:))';
            mahalanobis=(dist+dist1)/2;
            radii=mahalanobis*sqrt(diag(Spread(:,:,i)));
            for j=1:n 
if radii(j)<0.01
radii(j)=delta;
end
            end
      if model==1

        for j=1:n
        if z_current(j)<Center_lower(i,j)
                di_upper(i,j)=exp(-0.5*(z_current(j)-Center_lower(i,j))^(2)/radii(j)^(2));
            elseif z_current(j)>=Center_lower(i,j) & z_current(j)<=Center_upper(i,j)
                di_upper(i,j)=1;
            elseif z_current(j)>Center_upper(i,j)
                di_upper(i,j)=exp(-0.5*(z_current(j)-Center_upper(i,j))^(2)/radii(j)^(2));
        end
            if z_current(j)<=(Center_lower(i,j)+Center_upper(i,j))/2
                di_lower(i,j)=exp(-0.5*(z_current(j)-Center_upper(i,j))^(2)/radii(j)^(2));
            elseif z_current(j)>(Center_lower(i,j)+Center_upper(i,j))/2
                di_lower(i,j)=exp(-0.5*(z_current(j)-Center_lower(i,j))^(2)/radii(j)^(2));
            end
        end
         di1_upper(i)=prod(di_upper(i,:));
    di1_lower(i)=prod(di_lower(i,:));
      else
            %{
           dis_upper=abs(z_current(1:n)-Center_upper(i,:));
        dis1_upper=dis_upper./Spread(i,:);
                dis_lower=abs(z_current(1:n)-Center_lower(i,:));
        dis1_lower=dis_lower./Spread(i,:);
        %}
                di1_upper(i)=exp(-0.5*(z_current(1:n)-Center_upper(i,:))*Spread(:,:,i)*(z_current(1:n)-Center_upper(i,:))');
        di1_lower(i)=exp(-0.5*(z_current(1:n)-Center_lower(i,:))*Spread(:,:,i)*(z_current(1:n)-Center_lower(i,:))');
        end
  end
    for i=1:R-1
      for j=1:m
  % lambda(i,j)=(design_factor(j)*di1_upper(i)+(1-design_factor(j))*di1_lower(i))/(sum(di1_upper)+sum(di1_lower));
      lambda(i,j)=(design_factor(j)*di1_upper(i)+(1-design_factor(j))*di1_lower(i));
      end
    end
              tempat1=zeros(2,n);
     for i=1:n
 for j=1:2
     if i<=1
     if j<=1
     tempat1(j,i)=z_current(i);
     else
     tempat1(j,i)=2*z_current(i)*tempat1(j-1,i)-1;    
     end
     else
     if j<=1
 tempat1(j,i)=z_current(i);
     else
  tempat1(j,i)=2*z_current(i)*tempat1(j-1,i)-tempat1(j,i-1);
     end
     end
 end
     end
 %tempat2=tempat1';
 tempat3=zeros(1,2*n);
 tempat3(:)=tempat1;
  xek = [1, tempat3]';
    for i=1:R-1
        for j=1:m
            Psik((i-1)*((2*n)+1)+1:i*((2*n)+1),j) = lambda(i,j)/sum(lambda(:,j))*xek;    
      % Psik3((i-1)*((2*n)+1)+1:i*((2*n)+1),j) = di1_upper(i)*xek; 
        end
    end
 for i=1:R-1
            for j=1:m
                Tetak((i-1)*((2*n)+1)+1:i*((2*n)+1),j) = pik(:,j,i);
            end
 end
    % Prediction of model output for instant k+1
    for j=1:m
  %  y(k+1,j) = ((((1-design_factor(j))*(Psik(:,j)'*Tetak(:,j)))+(design_factor(j)*Psik3(:,j)'*Tetak(:,j)))/(sum(di1_upper)+sum(di1_lower)));    % eq.(23)
   y(k,j)=Psik(:,j)'*Tetak(:,j);
   residual(k,j)=z(k,n+j)-y(k,j);
    end
end
  end