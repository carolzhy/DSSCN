function [FR]=FRnew(z,Center_upper_new,Center_lower_new,Spread_new,design_factor,model,delta)
[N,nm]=size(z);
[~,n]=size(Center_upper_new);
m=nm-n;
FR=zeros(N,m);
for k=1:N
     z_current=z(k,:);
 dist=(z_current(1:n)-Center_lower_new)*Spread_new*(z_current(1:n)-Center_lower_new)';
            dist1=(z_current(1:n)-Center_upper_new)*Spread_new*(z_current(1:n)-Center_upper_new)';
            mahalanobis=(dist+dist1)/2;
            radii=mahalanobis*sqrt(diag(Spread_new));
            for j=1:n 
if radii(j)<0.01
radii(j)=delta;
end
            end

 di_upper=zeros(1,n);
    di_lower=zeros(1,n);
 if model==1

        for j=1:n
        if z_current(j)<Center_lower_new(j)
                di_upper(j)=exp(-0.5*(z_current(j)-Center_lower_new(j))^(2)/radii(j)^(2));
            elseif z_current(j)>=Center_lower_new(j) & z_current(j)<=Center_upper_new(j)
                di_upper(j)=1;
            elseif z_current(j)>Center_upper_new(j)
                di_upper(j)=exp(-0.5*(z_current(j)-Center_upper_new(j))^(2)/radii(j)^(2));
        end
            if z_current(j)<=(Center_lower_new(j)+Center_upper_new(j))/2
                di_lower(j)=exp(-0.5*(z_current(j)-Center_upper_new(j))^(2)/radii(j)^(2));
            elseif z_current(j)>(Center_lower_new(j)+Center_upper_new(j))/2
                di_lower(j)=exp(-0.5*(z_current(j)-Center_lower_new(j))^(2)/radii(j)^(2));
            end
        end
         di1_upper=prod(di_upper);
    di1_lower=prod(di_lower);
      else
            %{
           dis_upper=abs(z_current(1:n)-Center_upper(i,:));
        dis1_upper=dis_upper./Spread(i,:);
                dis_lower=abs(z_current(1:n)-Center_lower(i,:));
        dis1_lower=dis_lower./Spread(i,:);
        %}
                di1_upper=exp(-0.5*(z_current(1:n)-Center_upper_new)*Spread_new*(z_current(1:n)-Center_upper_new)');
        di1_lower=exp(-0.5*(z_current(1:n)-Center_lower_new)*Spread_new*(z_current(1:n)-Center_lower_new)');
 end
  for j=1:m
      FR(k,j)=(design_factor(j)*di1_upper+(1-design_factor(j))*di1_lower);
  end
end
end