
clc

input=normal_class(NEweatherdata);
sourcedata=[input NEweatherclass];
[nData,nData1]=size(sourcedata);
[creditcardoutput,pendigits_Data]=modify_dataset_zero_class(sourcedata);

data=[sourcedata(:,1:end-1) creditcardoutput];
chunk_size=500;
nFolds=round(nData/chunk_size);

%alpha=1;
alpha=0.001;
ninput=8;
noutput=2;
decreasingfactor=0.5;
threshold=0.005;
confidenceinterval=0.001;
model=1;
p1=1.5; %the active learning budget
p2=0.1;  %uncertainty factor
sample_deletion=0;%choose one to activate active learning strategy
subset=ninput;%the desired number of input attributes
RSMnew=0;
RSMdev=0;

ensembleoutput=[];
inputpruning=1;
input_pruning=0;
ensemblepruning1=1;
ensemblepruning2=1;
ensemblesize=[];
ensemblesamples=[];
A1=[];
B=[];
C=[];
D=[];
E=[];
F=[];
l=0;

buffer=[];
counter=0;
ensemble=0;
covarianceinput=ones(ninput,ninput);
for k=1:chunk_size:nData
    tic

lambdaD=min(1-exp(-counter/nFolds),0.09);
lambdaW=min(1-exp(-counter/(nFolds-1)),0.1);
  if (k+chunk_size-1) > nData
        Data = data(k:nData,:);    %Tn = T(n:nTrainingData,:);
        %Block = size(Pn,1);             %%%% correct the block size
       % clear V;                        %%%% correct the first dimention of V 
    else
        Data = data(k:(k+chunk_size-1),:);   % Tn = T(n:(n+Block-1),:);
  end
[r,q]=size(Data);

 
if ensemble==0
fix_the_model=r;
parameters(1)=p1; %the active learning budget
parameters(2)=p2; %uncertainty factor
Data_fix=Data;

[Center_upper,Center_lower,Spread, output_evo,time,rule,input_evo,count_samples,design_factor,classification_rate_testing,pik,focalpoints_upper,focalpoints_lower,sigmapoints,cik,population,population_class_cluster,weightinput,scope]= eSCN(Data_fix, fix_the_model,parameters,ninput,sample_deletion,input_pruning,subset);
weightinput=1;
buffer=Data;
[v,vv]=size(Center_upper);
network_parameters=2*v*subset+(subset)^(2)*v+(2*subset+1)*v*noutput;
network=struct('Center_upper',Center_upper,'Center_lower',Center_lower,'Spread',Spread,'ensemble_weight',rand(noutput,ninput),'design_factor',design_factor,'network_parameters',network_parameters,'fuzzy_rule',v,'CR',classification_rate_testing,'pik',pik,'focalpoints_upper',focalpoints_upper,'focalpoints_lower',focalpoints_lower,'sigmapoints',sigmapoints,'cik',cik,'population',population,'population_class_cluster',population_class_cluster,'weightinput',weightinput,'born',counter,'scope',scope);
ensemble=ensemble+1;
ensemblesize(1)=1;
ensemblesamples(1)=count_samples;
error(:,1,1)=0;
for k3=1:noutput
covariance(1,:,k3)=0;
covariance(:,1,k3)=0;
end
covariance_old=covariance;
else
    counter=counter+1;
    Datatest=Data;
ensembleoutputtest=zeros(size(Datatest,1),1);
individualoutputtest=zeros(size(Datatest,1),size(network,1));
misclassification=0;
outens=[];
for k1=1:size(Datatest,1)
    stream=Datatest(k1,:);
    % output=zeros(1,noutput);
 for m=1:ensemble
              weighted_stream=stream(1:ninput).*network(m).weightinput;
               if m>1
            weighted_stream=weighted_stream+alpha*ysem*network(m).ensemble_weight;
               end
            xek = [1, weighted_stream]';
        [nrule,ndimension]=size(network(m).Center_upper);
     
                    input=[weighted_stream stream(ninput+1:end)];
                    
            [ysem]=inference(input,network(m).Center_upper,network(m).Center_lower,network(m).Spread,network(m).design_factor,network(m).pik,model,p1);
         
      
 end
 clear weightperrule 
 [maxout1,trueclasslabel]=max(stream(ndimension+1:end));
      [maxout,ensemblelabel]=max(ysem);
        ensembleoutputtest(k1)=ensemblelabel;
        if trueclasslabel==ensemblelabel
            misclassification=misclassification+1;
        end
end
totalrule=0;
totalparameters=0;
for m=1:size(network,1)
    totalrule=totalrule+network(m).fuzzy_rule;
    totalparameters=totalparameters+network(m).network_parameters;
end
time=toc;

B(counter)=(misclassification)/size(Datatest,1);
C(counter)=totalrule;
D(counter)=totalparameters;
%

E(counter)=size(network,1);
inputexpectation=mean(Data(:,1:ninput));
 inputvariance=var(Data(:,1:ninput));
 temporary=zeros(chunk_size,ninput);
 [upperbound,upperboundlocation]=max(Data(:,1:ninput));
 [lowerbound,lowerboundlocation]=min(Data(:,1:ninput));
 for iter=1:size(Data,1)
     for iter1=1:ninput
     temporary(iter,iter1)=Data(iter,iter1)-inputexpectation(iter1);
     end
 end
 inputcovar=zeros(ninput,ninput);
 covarianceinput_old=covarianceinput;
      for i=1:ninput
                           for o=1:ninput
                               if i~=o
                           temporary1=cov(Data(:,i),Data(:,o));
                           inputcovar(i,o)=temporary1(1,2);
                            covarianceinput(i,o)=(covarianceinput_old(i,o)*(counter-1)+(((counter-1)/counter)*inputcovar(i,o)))/counter;
                               end
                           end
      end
      if inputpruning==1
            FCorrelation=zeros(ninput,ninput);
                       input_weight=zeros(1,ninput);
                  
                       for i=1:ninput
                       for o=1:ninput
                       if i~=o
                           pearson=covarianceinput(i,o)/sqrt(covarianceinput(o,o)*covarianceinput(i,i));
            FCorrelation(i,o)=(0.5*(covarianceinput(i,i)+covarianceinput(o,o))-sqrt((covarianceinput(o,o)+covarianceinput(i,i))^(2)-4*covarianceinput(i,i)*covarianceinput(o,o)*(1-pearson^(2))));
                       end
                       end
                       end
                      
                                              for i=1:ninput
                       input_weight(i)=mean((FCorrelation(i,:)));
                                              end
                                  
                       input_weight_display=(input_weight/max(input_weight));
                                 
                       for i=1:ninput
                               history(counter,i)=input_weight_display(i);
                       end
                       Data(:,1:ninput)=Data(:,1:ninput).*input_weight_display;
      end
    covariance_old=covariance;
      storeoutput=[]; 
  traceinputweight=[];   
    for k1=1:size(Data,1)
                stream=Data(k1,:);
    
                        xek = [1, stream(1:ninput)]';
                        weight_input1=ones(1,ninput);
                     
                            output=zeros(1,noutput);
                            pruning_list=[];
        for m=1:ensemble
            weighted_stream=stream(1:ninput).*network(m).weightinput;
            if m>1
            weighted_stream=weighted_stream+alpha*ysem*network(m).ensemble_weight;
            end
            
            xek = [1, weighted_stream]';
        [nrule,ndimension]=size(network(m).Center_upper);
      
            input=[weighted_stream stream(ninput+1:end)];
            [ysem]=inference(input,network(m).Center_upper,network(m).Center_lower,network(m).Spread,network(m).design_factor,network(m).pik,model,p1);
          
       
         [maxout,classlabel]=max(ysem);
    [maxout1,trueclasslabel]=max(stream(ndimension+1:end));
    if classlabel~=trueclasslabel
        error(k1,1,m)=1;
    else
        error(k1,1,m)=0;
    end
   
        for iter=1:noutput
        storeoutput(k1,iter,m)=ysem(iter);
        end
    clear Psik2
        end
               clear weightperrule 
        [maxout,ensemblelabel]=max(ysem);
        ensembleoutput(k1)=ensemblelabel;
        ensemblesize(k1)=ensemble;
  
        pruning_list=[];
           
            outputcovar=zeros(ensemble,ensemble,noutput);
           
            if k1==size(Data,1) && ensemblepruning2==1
                        for iter=1:ensemble
      
                        for iter1=1:ensemble
                            for iter2=1:noutput
                            temporary=cov(storeoutput(:,iter2,iter1),storeoutput(:,iter2,iter));
                        outputcovar(iter,iter1,iter2)=temporary(1,2);
                        covariance(iter,iter1,iter2)=(covariance_old(iter,iter1,iter2)*(counter-1)+(((counter-1)/counter)*outputcovar(iter,iter1,iter2)))/counter;
                            end
                        end
                        end
            end
        if ensemblepruning2==1 && ensemble>1  && k1==size(Data,1)
                       merged_list=[];
                      
                for l=0:ensemble-2
        for hh=1:ensemble-l-1
            MCI=zeros(1,noutput);
            for o=1:noutput
            pearson=covariance(end-l,hh,o)/sqrt(covariance(end-l,end-l,o)*covariance(hh,hh,o));
            MCI(o)=(0.5*(covariance(hh,hh,o)+covariance(end-l,end-l,o))-sqrt((covariance(hh,hh,o)+covariance(end-l,end-l,o))^(2)-4*covariance(end-l,end-l,o)*covariance(hh,hh,o)*(1-pearson^(2))));
            end
       
                                           if max(abs(MCI))<threshold %&& counter-network(end-l).born>5 && counter-network(hh).born>5 %(max(MCI)<0.1 & max(MCI)>0) & (max(MCI)>-0.1 & max(MCI)<0)
           if isempty(merged_list)
          merged_list(1,1)=ensemble-l;
          merged_list(1,2)=hh;
          else
            No=find(merged_list(:,1:end-1)==ensemble-l);
            No1=find(merged_list(:,1:end-1)==hh);
            if isempty(No) && isempty(No1)
          merged_list(end+1,1)=ensemble-l;
          merged_list(end+1,2)=hh;
            end
           end
           break
                               end 
        end
                end
            
                                        del_list=[];
                                          misclassification=zeros(1,size(network,1));
for m=1:size(network,1)
%for out=1:noutput
misclassification(m)=sum(error(:,1,m));
%end
end
                    for i=1:size(merged_list,1)
                    No2=find(merged_list(i,:)==0);
                    if isempty(No2)
                                            if misclassification(merged_list(i,1))>misclassification(merged_list(i,2))
                      a=merged_list(i,1);
                      b=merged_list(i,2);
                      else
                        b=merged_list(i,1);
                      a=merged_list(i,2);    
                                            end
                                              del_list=[del_list b];                     
                    end
                    end
                       if isempty(del_list)==false
                 network(del_list,:)=[];
        error(:,:,del_list)=[];
        ensemble=size(network,1);
        ensemblesize(k1)=ensemble;
                covariance(:,pruning_list,:)=[];
        covariance(pruning_list,:,:)=[];
        covariance_old=covariance;
                       end
            
        end
        if k1==size(Data,1)
        Zstat=mean(Data(:,1:ninput));
    cuttingpoint=0;
        

        for cut=1:size(Data,1)
        Xstat=mean(Data(1:cut,1:ninput));
        [Xupper,Xupper1]=max(Data(1:cut,1:ninput));
        [Xlower,Xlower1]=min(Data(1:cut,1:ninput));
        Xbound=(Xupper-Xlower)*sqrt(((r-cut)/(2*cut*(r))*reallog(1/confidenceinterval)));
        Zbound=(upperbound-lowerbound).*sqrt(((r-cut)/(2*cut*(r))*reallog(1/confidenceinterval)));
        if mean(Xbound+Xstat)>=mean(Zstat+Zbound) && cut<r
            cuttingpoint=cut;
              Ystat=mean(Data(cuttingpoint+1:end,1:ninput));
                      [Yupper,Yupper1]=max(Data(cuttingpoint+1:end,1:ninput));
        [Ylower,Ylower1]=min(Data(cuttingpoint+1:end,1:ninput));
         Ybound=(Yupper-Ylower).*sqrt(((r-cuttingpoint)/(2*cuttingpoint*(r-cuttingpoint)))*reallog(1/lambdaD));
          Ybound1=(Yupper-Ylower).*sqrt(((r-cuttingpoint)/(2*cuttingpoint*(r-cuttingpoint)))*reallog(1/lambdaW));
            break
       
        end
        end
if cuttingpoint==0
Ystat=Zstat;  
            Ybound=(upperbound-lowerbound).*sqrt(((r-cut)/(2*cut*(r))*reallog(1/lambdaD)));
            Ybound1=(upperbound-lowerbound).*sqrt(((r-cut)/(2*cut*(r))*reallog(1/lambdaW)));
end

         if (mean(abs(Xstat-Ystat)))>=mean(Ybound)
            %% drift        
if isempty(buffer)
Data_fix=Data;
 fix_the_model=r;
else
   Data_fix=[Data;buffer]; 
    fix_the_model=size(Data,1)+size(buffer,1);
end
%
store_Z=rand(noutput,ninput);
for k3=1:size(Data,1)
Data_fix(k3,1:ninput)=Data_fix(k3,1:ninput)+alpha*storeoutput(k3,:,ensemble)*store_Z;
end
parameters(1)=p1; %the active learning budget
parameters(2)=p2; %uncertainty factor
[Center_upper,Center_lower,Spread, output_evo,time,rule,input_evo,count_samples,design_factor,classification_rate_testing,pik,focalpoints_upper,focalpoints_lower,sigmapoints,cik,population,population_class_cluster,weightinput,scope]= eSCN(Data_fix, fix_the_model,parameters,ninput,sample_deletion,input_pruning,subset);
weightinput=1;
[v,vv]=size(Center_upper);
network_parameters=2*v*subset+(subset)^(2)*v+(2*subset+1)*v*noutput;
network=[network; struct('Center_upper',Center_upper,'Center_lower',Center_lower,'Spread',Spread,'ensemble_weight',rand(noutput,ninput),'design_factor',design_factor,'network_parameters',network_parameters,'fuzzy_rule',v,'CR',classification_rate_testing,'pik',pik,'focalpoints_upper',focalpoints_upper,'focalpoints_lower',focalpoints_lower,'sigmapoints',sigmapoints,'cik',cik,'population',population,'population_class_cluster',population_class_cluster,'weightinput',weightinput,'born',counter,'scope',scope)];
ensemble=size(network,1);
ensemblesize(k1)=ensemble;
ensemblesamples(counter)=count_samples;
     
for k3=1:noutput
error(:,k3,ensemble)=0;
covariance(:,ensemble,k3)=0;
covariance(ensemble,:,k3)=0;
end
buffer=[];
        elseif (mean(abs(Xstat-Ystat)))>=mean(Ybound1) && (mean(abs(Xstat-Ystat)))<mean(Ybound)
            %% Warning
            buffer=[buffer;Data];
          
        else
            %%stable
              misclassification=zeros(1,size(network,1));
for m=1:size(network,1)
% for out=1:noutput
misclassification(m)=sum(error(:,1,m));
% end
end

[Rselected,index1]=min(misclassification);

  fix_the_model=r;
parameters(1)=p1; %the active learning budget
parameters(2)=p2; %uncertainty factor

Data_fix=Data;
buffer=[];
SC=zeros(1,ensemble);
%for i=1:ensemble
if ensemble>1
for k3=1:size(Data,1)
Data_fix(k3,1:ninput)=Data_fix(k3,1:ninput)+alpha*storeoutput(k3,:,end-1)*network(end).ensemble_weight;
end
end
%}
store_Z=network(end).ensemble_weight;

[Center_upper,Center_lower,Spread, output_evo,time,rule,input_evo,count_samples,design_factor,classification_rate_testing,pik,focalpoints_upper,focalpoints_lower,sigmapoints,cik,population,population_class_cluster,weightinput,scope]= eSCNupdate(Data_fix, fix_the_model,parameters,ninput,sample_deletion,input_pruning,subset,network(end));
weightinput=1;
[v,vv]=size(Center_upper);
network_parameters=2*v*subset+(subset)^(2)*v+(2*subset+1)*v*noutput;
SC(end)=count_samples;
replacement=struct('Center_upper',Center_upper,'Center_lower',Center_lower,'Spread',Spread,'ensemble_weight',rand(noutput,ninput),'design_factor',design_factor,'network_parameters',network_parameters,'fuzzy_rule',v,'CR',classification_rate_testing,'pik',pik,'focalpoints_upper',focalpoints_upper,'focalpoints_lower',focalpoints_lower,'sigmapoints',sigmapoints,'cik',cik,'population',population,'population_class_cluster',population_class_cluster,'weightinput',weightinput,'born',network(end).born,'scope',network(end).scope);    
network(end)=replacement;
%end
        ensemblesamples(counter)=mean(SC);

        end
    end
   % end
    % % Start the model evolution (learning and prediction)
    end
    H(counter)=time;
end




end
Brat=mean(B);
Bdev=std(B);
Crat=mean(C);
Cdev=std(C);
Drat=mean(D);
Ddev=std(D);
%
Erat=mean(E);
Edev=std(E);

Hrat=mean(H);
Hdev=std(H);


