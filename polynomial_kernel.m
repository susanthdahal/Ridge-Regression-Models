function [avg_error]=polynomial_kernel(data,label,s_no)
    a=[-1,-0.5, 0, 0.5, 1];
    c=[1, 2, 3, 4];
        minerr=1;
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
        opt_lamda=100000;
        opt_a=10000;
        opt_c=10000;
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
            minerr1=1;
            for count1=1:size(a,2)
                minerr2=1;
                for count2=1:size(c,2)
                    K=(train*train'+ a(count1)).^c(count2); 
                    k=(train*test'+ a(count1)).^c(count2); 
                    inverse=pinv(K + lamda(lam)*eye(size(train,1),size(train,1)));
                    y=((train_label'*inverse)* k)';
                    err2(count2)=mean(power((y-test_label),2));
                    mincheck=err2(count2);
                    if mincheck<minerr2
                        minerr2=mincheck;
                        opt_c=c(count2);
                    end
                end
                err1(count1)=mean(err2);
                if mincheck<minerr1
                    minerr1=mincheck;
                    opt_a=a(count1);
                end
            end
            errj(j)=mean(err1);
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
        fprintf('\nSplit #%d\t opt Lambda=%f  a=%f  c=%d ',s_no,opt_lamda,opt_a,opt_c);
        train=tempx;
        test= testx;
        train_label=templ;
        test_label=testl;
        K=(train*train'+ opt_a).^opt_c; 
        inverse=pinv(K + (opt_lamda*eye(size(train,1),size(train,1))));
        k=(train*test'+ opt_a).^opt_c;
        y=((train_label'*inverse)* k)';
        avg_error=mean(power((y-test_label),2));
end