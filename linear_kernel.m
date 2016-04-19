function [avg_error]=linear_kernel(data,label,s_no)
        length=size(data,1);
        breadth=size(data,2);
        r=randperm(length);
        r=r';
        temp=data(r,:);
        label=label(r);
        breakpoint=floor(length*50/100);
        tr=1:breakpoint;
        tr=tr';
        tempx=temp(tr,:);
        templ=label(tr);
        te=(breakpoint+1):length;
        te=te';
        testx=temp(te,:);
        testl=label(te);
        [tempx,mu,sigma]=zscore(tempx);
        for l=1:breadth
            testx(:,l)=(testx(:,l)-mu(l))/sigma(l);
        end
        breakpoint_values=linspace(1,breakpoint,6);
        breakpoint_values=ceil(breakpoint_values);
        lamda=[0 1e-4 1e-3 1e-2 1e-1 1 1e+1 1e+3 1e+3];
        error=[];
        minerr=1;
        for k=1:size(lamda,2)
            err=[];
            for j=1:5
                r=breakpoint_values(j):breakpoint_values(j+1);
                r=r';
                test=tempx(r,:);
                test_label=templ(r,:);
                train=tempx;
                train_label=templ;
                train(r,:)=[];
                train_label(r,:)=[];
                [train,mu,sigma]=zscore(train);
                for l=1:breadth
                    test(:,l)=(test(:,l)-mu(l))/sigma(l);
                end
                inverse=pinv(train*train' +lamda(k)*eye(size(train,1),size(train,1)));
                y=((train_label'*inverse)* (train*test'))';
                err(j)=mean(power((y-test_label),2));
            end
                if mean(err)<minerr
                        minerr=mean(err);
                        opt_lamda=lamda(k);
                end
        end
        fprintf('\nSplit #%d\t opt Lambda=%f',s_no,opt_lamda);
        train=[ones(size(tempx,1),1) tempx];
        test=[ones(size(testx,1),1) testx];
        train_label=templ;
        test_label=testl;
        inverse=pinv(train*train' +opt_lamda*eye(size(train,1),size(train,1)));
        y=((train_label'*inverse)* (train*test'))';
        avg_error=mean(power((y-test_label),2));
end