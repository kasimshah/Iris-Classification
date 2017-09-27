clear
clc

avgAccur=0.0;
data=dlmread('iris_dataset.dat');
numrows=size(data,1); %Get the number of rows for data
shuff=data(randperm(numrows),:);%Shuffe the data set and store it

%For loop to allow for cross validation and determine the different
%training/test data for each iteration
for iteration=1:10
    
    shuffdata=shuff; %Store shuff into shuffdata to be used later on and manipulated with
    
    cutoff=.1*numrows; %Determine how many rows to get for testing
    start=((iteration-1)*cutoff)+1;
    finish=start+cutoff-1;
    
    testData=shuffdata([start:finish],:); %Create test data  
    shuffdata([start:finish],:)=[]; %Take away the data data block from shuffdata
    trainingData=shuffdata; %Create training data
    
    classTraining=sortrows(trainingData,5); %Training data sorted based on class
    rowsClassTraining=size(classTraining,1); %Get the number of rows for classTraining
    
    gc=classTraining(:,5); %group of classes
    c=findgroups(gc); %find the different group of classes 
    splitapply(@mean,classTraining,c); %find the means of the features based on the classes
    
    %Split the means into three different vectors for each class
    m=splitapply(@mean,classTraining,c);
    mean1=m(1,(1:4));
    mean2=m(2,(1:4));
    mean3=m(3,(1:4));
    
    sumcov1=[0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0];
    sumcov2=[0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0];
    sumcov3=[0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0];
    total1=0;
    total2=0;
    total3=0;
    
    %For loop to calculate sum to be used for the covriance matrix for each
    %class
    for i=1:rowsClassTraining
        
        if classTraining(i,5)==1.0
            trainEx=classTraining(i,(1:4));
            subVec1=trainEx'- mean1';
            sumcov1=sumcov1+(subVec1*subVec1');
            total1=total1+1;
            
        elseif classTraining(i,5)==2.0
            trainEx=classTraining(i,(1:4));
            subVec2=trainEx'- mean2';
            sumcov2=sumcov2+(subVec2*subVec2');
            total2=total2+1;
            
        else
            trainEx=classTraining(i,(1:4));
            subVec3=trainEx'- mean3';
            sumcov3=sumcov3+(subVec3*subVec3');
            total3=total3+1;
        end
        
    end
    
    %Caluclate covariance for the three classes
    cov1=sumcov1/total1;
    cov2=sumcov2/total2;
    cov3=sumcov3/total3;
    
    rowsTestData=size(testData,1);
    
    right=0;
    %For loop to be used with each test data sample to determine which
    %class it belongs to and see if the right predication made
    for i=1:rowsTestData
        
        testsam=testData(i,(1:4));
        
        gsubvec1=testsam'-mean1';
        gsubvec2=testsam'-mean2';
        gsubvec3=testsam'-mean3';
        
        icov1=inv(cov1);
        icov2=inv(cov2);
        icov3=inv(cov3);
        
        detcov1=.5*log(det(cov1));
        detcov2=.5*log(det(cov2));
        detcov3=.5*log(det(cov3));
        
        g1= -.5*gsubvec1'*icov1*gsubvec1-detcov1+log(1/3);
        g2= -.5*gsubvec2'*icov2*gsubvec2-detcov2+log(1/3);
        g3= -.5*gsubvec3'*icov3*gsubvec3-detcov3+log(1/3);
        
        if g1>g2 && g1>g3
            class=1.0;
        elseif g2>g1 && g2>g3
            class=2.0;
        else
            class=3.0;
        end
        
        if class==testData(i,5)
            right=right+1;
        end
        
    end
    
    accurate=(right/rowsTestData)*100;
    
    avgAccur=avgAccur+accurate;
    
    fprintf('The accuracy for iteration %i is %.2f\n',iteration,accurate)
    
    clear shuffdata testData trainingData classTraining gc c m mean1 mean2 mean3 sumcov1 sumcov2 sumcov3
    
end

fprintf('The average accuracy for the 10 iterations of CV is %.2f\n',avgAccur/10.0)

