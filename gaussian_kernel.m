function [avg_error]=gaussian_kernel(data,label,s_no)

        length=size(data,1);
        breadth=size(data,2);
        
        opt_lamda=10000;
        opt_sigma=10000;
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
        sig=[0.125 0.25 0.5 1 2 4 8];
        error=[];
        for lam=1:size(lamda,2)
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
            minerr2=1;
            for count1=1:size(sig,2)
                K=calc_k(train,train,sig(count1));
                inverse=pinv(K +lamda(lam)*eye(size(train,1),size(train,1)));
                y=((train_label'*inverse)* calc_k(train,test,sig(count1)))';
                err1(count1)=mean(power((y-test_label),2));
                mincheck=err1(count1);
                if mincheck<minerr2
                        minerr2=mincheck;
                        opt_sigma=sig(count1);
                end
            end
            err(j)=mean(err1);
        end
        error(lam)=mean(err);
        if lam==1
            minerr= error(lam);
            opt_lamda=lamda(lam);
        end
        if error(lam) < minerr
            minerr=error(lam);
            opt_lamda=lamda(lam);
        end
        end
        fprintf('\nSplit #%d\t opt Lambda=%f sigma=%f',s_no,opt_lamda,opt_sigma);
        train=[ones(size(tempx,1),1) tempx];
        test=[ones(size(testx,1),1) testx];
        train_label=templ;
        test_label=testl;
        K=calc_k(train,train,opt_sigma);
        inverse=pinv(K +opt_lamda*eye(size(train,1),size(train,1)));
        y=((train_label'*inverse)* calc_k(train,test,opt_sigma))';
        avg_error=mean(power((y-test_label),2));
end
function [mat]=calc_k(train,test,s)
    for i=1:size(train,1)
        for j=1:size(test,1)
           temp=train(i,:)-test(j,:);
           n=norm(temp);
           mat(i,j)=exp(-n/(s*s));
        end
    end
end