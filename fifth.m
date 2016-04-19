function fifth()
    datasets=-1+2*rand(100,10);
    y_with_noise=2*power(datasets,2)+ normrnd(0,sqrt(.1));
    y=2*power(datasets,2);
    v=zeros(100,6);
    b=zeros(100,6);
    mse=zeros(100,6);
    for i=1:100
        temp=[];
         for j=1:6  
            if j==1
                weights{i,j}=1;
                y_pred=ones(10,1);
            else
                temp=[temp,power(datasets(i,:)',j-2)];
                w= (pinv(temp'*temp))*temp'*y_with_noise(i,:)';
                weights{i,j}=w;
                y_pred=temp*w;
            end
            pred{i,j}=y_pred';
            mse(i,j)=mean(power((y_with_noise(i,:)'-y_pred),2));
        end
    end
    for i=1:100
        temp=[];
         for j=1:6  
            if j==1
                y_hat=ones(10,1);
            else
                temp=[temp,power(datasets(i,:)',j-2)];
                mat=cell2mat(weights(:,j)');
                w=mean(mat');
                y_hat=temp*w';
            end
            b(i,j)=mean(power((y(i,:)'-y_hat),2));
            mat=cell2mat(pred(i,j));
            v(i,j)=var(y_hat);
         end    
    end
    name='5a g';
    for i=1:6
        figure('Name',strcat(name,int2str(i)),'NumberTitle','off');
        hist(mse(:,i),10);
        fprintf('\nFor g%d bias=%f  and  variance is %f',i,mean(b(:,i)),mean(v(:,i)));
    end
    
    
    
    
    fprintf('\n\n Fifth b part \n\n');
    datasets=-1+2*rand(100,100);
    y_with_noise=2*power(datasets,2)+ normrnd(0,sqrt(.1));
    y=2*power(datasets,2);
    v=zeros(100,6);
    b=zeros(100,6);
    mse=zeros(100,6);
    for i=1:100
        temp=[];
         for j=1:6  
            if j==1
                weights{i,j}=1;
                y_pred=ones(100,1);
            else
                temp=[temp,power(datasets(i,:)',j-2)];
                w= (pinv(temp'*temp))*temp'*y_with_noise(i,:)';
                weights{i,j}=w;
                y_pred=temp*w;
            end
            pred{i,j}=y_pred';
            mse(i,j)=mean(power((y_with_noise(i,:)'-y_pred),2));
        end
    end
    for i=1:100
        temp=[];
         for j=1:6  
            if j==1
                y_hat=ones(100,1);
            else
                temp=[temp,power(datasets(i,:)',j-2)];
                mat=cell2mat(weights(:,j)');
                w=mean(mat');
                y_hat=temp*w';
            end
            b(i,j)=mean(power((y(i,:)'-y_hat),2));
            mat=cell2mat(pred(i,j));
            v(i,j)=var(y_hat);
         end    
    end
    name='5b g';
    for i=1:6
        figure('Name',strcat(name,int2str(i)),'NumberTitle','off');
        hist(mse(:,i),10);
        fprintf('\nFor g%d bias=%f  and  variance is %f',i,mean(b(:,i)),mean(v(:,i)));
    end
    fprintf('\n');
end